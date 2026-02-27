# -*- coding: utf-8 -*-
"""
Visual Log Extraction Protocol for A-MoE Framework
This script utilizes a Vision-Language Model (VLM) via the ms-swift framework to process
raw video data. It extracts keyframes and sequentially analyzes them to construct a detailed
textual log of the agent's egocentric observations and kinematic actions.
"""

import argparse
import base64
import datetime
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import cv2  # pip install opencv-python
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from tqdm import tqdm

from scripts.analysis.keyframe_utils import extract_keyframes_from_video

# ==========================================
# Framework Dependencies (ms-swift)
# ==========================================
try:
    from swift.llm import (
        InferRequest,
        PtEngine,
        RequestConfig,
        get_model_tokenizer,
        get_template
    )
    from swift.llm.template import load_file, load_image
    from swift.tuners import Swift
except ImportError:
    print("❌ Critical Error: The 'ms-swift' framework is required but not found.")
    print("Please install it using: pip install ms-swift")
    sys.exit(1)

# ==========================================
# Global Setup & Logging
# ==========================================
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_visual_logs.log", mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# ==========================================
# Text Processing Utilities
# ==========================================
def extract_qa(text: str) -> str:
    """Extracts the core question part from a formatted text block."""
    index = text.find("Question:")
    if index != -1:
        result = text[index + len("Question:"):].strip()
        return result
    return text

def split_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the vision model's multi-point output (action and view state).
    Splits the numbered lists ('1.', '2.', '3.') into structured columns.
    """
    def extract_points(text):
        try:
            if '1.' in text and '2.' in text and '3.' in text:
                point_1_match = re.search(r'(?:###\s*1\.|1\.)(.*?)(?=\n\s*(?:###\s*2\.|2\.))', text, re.DOTALL)
                point_2_match = re.search(r'(?:###\s*2\.|2\.)(.*?)(?=\n\s*(?:###\s*3\.|3\.))', text, re.DOTALL)
                point_3_match = re.search(r'(?:###\s*3\.|3\.)(.*)', text, re.DOTALL)

                point_1 = point_1_match.group(1).strip() if point_1_match else None
                point_2 = point_2_match.group(1).strip() if point_2_match else None
                point_3 = point_3_match.group(1).strip() if point_3_match else None

                if point_3 is None:
                    view_content = point_2
                else:
                    view_content = str(point_2) + ' ' + str(point_3)
                return point_1, view_content
            else:
                return None, text
        except Exception:
            return None, text

    df[['action', 'view']] = df.iloc[1:, 0].apply(lambda x: pd.Series(extract_points(str(x))))
    df.loc[0, 'view'] = df.iloc[0, 0]
    return df

def derive_video_content_str(df: pd.DataFrame) -> str:
    """
    Synthesizes the parsed frame-by-frame dataframe into a continuous, 
    coherent textual log representing the temporal sequence.
    """
    video_content_str = (
        "The content of the video will be described in the form of text. "
        "The 0th frame is the initial frame, which includes the scene that agent observe at the initial position. "
        "The agent keeps moving, thus constantly obtaining new visual observations (frames). "
        "The last frame is the visual observation at the current position."
    )
    video_content_str += f"\nFrame 0: Observe: {df['view'].iloc[0]}"
    for i in range(1, df.shape[0]):
        video_content_str += f"\nFrame {i}: After {df['action'].iloc[i]}, Observe: {df['view'].iloc[i]}"

    return video_content_str

def question_fil(text: str) -> str:
    """Removes the 'Question:' prefix if present."""
    start_index = text.find("Question:")
    if start_index != -1:
        result = text[start_index + len("Question:"):]
        return result
    return text

# Regex patterns for accuracy calculation utility
BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
ANSWER_PATTERN = r"<answer>\s*([A-H])"
SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\." 
SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])" 

def extract_option_letter(answer: str) -> str:
    """Extracts the selected option (A-H) from the final string output."""
    answer = str(answer).strip()
    answer_matches = list(re.finditer(ANSWER_PATTERN, answer, re.IGNORECASE))
    if answer_matches:
        return answer_matches[0].group(1).upper()

    dot_matches = list(re.finditer(SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
    if dot_matches:
        return dot_matches[-1].group(1).upper()

    return answer.upper()

# ==========================================
# Core Perception Engine
# ==========================================
class VideoProcessor:
    """
    Video processing class. Responsible for extracting video frames, encoding them,
    and querying the local Vision-Language Model to build chronological semantic logs.
    """

    def __init__(self, vision_model_path: str):
        log.info(f"Loading visual parsing model: {vision_model_path}")
        self.model, self.tokenizer = get_model_tokenizer(
            vision_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.template = get_template("qwen2_5_vl", self.tokenizer)
        self.engine = PtEngine.from_model_template(self.model, self.template)
        log.info("✅ Visual parsing model initialized successfully.")

    def extract_frames(self, video_path: str, max_frames: int = None):
        """Extracts frames from a video file and encodes them in base64."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        base64_frames = []

        # Read all frames first
        all_read_frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            all_read_frames.append(frame)
        cap.release()

        # Uniformly sample frames if max_frames is set to prevent memory overload
        if max_frames and len(all_read_frames) > max_frames:
            indices = np.linspace(0, len(all_read_frames) - 1, max_frames, dtype=int)
            sampled_frames = [all_read_frames[i] for i in indices]
        else:
            sampled_frames = all_read_frames

        for frame in sampled_frames:
            frames.append(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        return frames, base64_frames

    def analyze_video(self, video_path: str, qa: str = None, max_frames: int = 32):
        """
        Analyzes a sequence of video frames to extract relative movement and spatial context.
        Operates in a sliding dual-frame contrast mode.
        """
        _, base64_frames = self.extract_frames(video_path, max_frames=max_frames)
        res = pd.DataFrame(columns=['description'])

        for frame_idx in range(len(base64_frames)):
            try:
                if frame_idx > 0:
                    # Sequential Differential Scan Prompt
                    prompt = (
                        "I provide you with the agent's first-person perspective. Two images represent the field of view before and after the action. Please output: \n"
                        "1. Based on the relative positional changes of objects in the field of view, determine the action performed (only output one of the following options without redundant content): Move forward, move backward, move left, move right, move upward, move downward, rotate left, rotate right, tilt downward, or tilt upward. \n"
                        "2. Analyze the post-action field of view compared to the pre-action view, identifying any newly captured spatial information. This includes objects that were previously invisible or unclear. Note that many objects of the same category may appear repetitive, such as buildings in a city, but they might differ in color or shape. When describing these, include features such as color and shape. Additionally, focus on the relationship between the agent and its surrounding environment. To do so, first imagine the three-dimensional scene around the agent. When describing relative positions, use terms such as 'to the left,' 'in the front-left,' or 'below' rather than simply referring to their positions in the field of view. \n"
                        f"3. If the objects mentioned in the following question appear in the images, please make sure to describe them: '{qa}'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n"
                        "Ensure responses are concise and clear."
                    )
                    content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frames[frame_idx-1]}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frames[frame_idx]}"}}
                    ]
                else:
                    # Initial Frame Prompt
                    prompt = (
                        "I provide you with the first-person perspective of an intelligent agent. "
                        "Please output a description of the current observed scene. "
                        "When describing the scene, focus on using features such as color and shape to characterize the objects. "
                        "Additionally, emphasize the spatial relationship between the agent itself and the surrounding environment. "
                        "You should first visualize the three-dimensional space around the agent. "
                        "When describing relative positions, use terms such as 'to the left,' 'ahead-left,' or 'below,' rather than merely stating their positions within the field of view. "
                        f"More important, if the objects mentioned in the following question appear in the images, please make sure to describe them: '{qa}'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n"
                        "The response should be concise and clear."
                    )
                    content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frames[frame_idx]}"}}
                    ]

                messages = [{"role": "user", "content": content}]
                infer_request = InferRequest(messages=messages)
                response = self.engine.infer([infer_request])[0]

                frame_description = response.choices[0].message.content
                res.loc[frame_idx, 'description'] = frame_description

            except Exception as e:
                # Use tqdm.write to avoid breaking the progress bar display
                tqdm.write(f"⚠️ An error occurred at frame {frame_idx}: {e}")
                res.loc[frame_idx, 'description'] = None
                time.sleep(5)  # Brief pause to recover from potential API/VRAM throttling

        return res

# ==========================================
# Main Execution Protocol
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Perception Engine: Extract textual semantic logs from raw videos via VLM.')
    parser.add_argument('--data_paths', type=str, nargs='+', default=['dataset/complete/test_data.json'], help='List of JSON files containing dataset metadata')
    parser.add_argument('--folder_path', type=str, default='dataset/complete/videos', help='Root directory containing the raw mp4/avi videos')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='HuggingFace path to the Vision Model')
    parser.add_argument('--save_path', type=str, default='results/inter', help='Directory to output the generated textual logs')
    parser.add_argument('--max_frames', type=int, default=32, help='Maximum number of frames to sample per video to manage memory constraints')
    args = parser.parse_args()

    # Initialize Perception Engine
    video_processor = VideoProcessor(args.model_path)
    
    # Establish output directories
    video_keyframe_folder = os.path.join(args.save_path, 'video_keyframe')
    os.makedirs(video_keyframe_folder, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    for data_path in args.data_paths:
        if not os.path.exists(data_path):
            log.warning(f"File not found: {data_path}. Skipping.")
            continue

        save_path = os.path.join(args.save_path, f'processed_{os.path.basename(data_path)}')

        # Resumption Protocol
        if os.path.exists(save_path):
            log.info(f"Found existing progress file. Resuming from: {save_path}")
            qa_df = pd.read_json(save_path)
            if 'processed' not in qa_df.columns:
                qa_df['processed'] = False
        else:
            log.info(f"Starting fresh processing for dataset: {data_path}")
            qa_df = pd.read_json(data_path)
            qa_df['response'] = None
            qa_df['processed'] = False

        pbar = tqdm(range(qa_df.shape[0]), desc=f"Processing {os.path.basename(data_path)}", unit="video")
        
        for idx in pbar:
            if qa_df.loc[idx, 'processed']:
                continue

            start_time = time.time()
            video_name = qa_df.loc[idx, 'video_id']
            qa = extract_qa(str(qa_df.loc[idx, 'question']))

            video_path = os.path.join(video_keyframe_folder, video_name)
            
            # Keyframe Extraction Check
            if not os.path.exists(video_path):
                original_video_path = os.path.join(args.folder_path, video_name)
                if not os.path.exists(original_video_path):
                    tqdm.write(f"⚠️ Original video not found: {original_video_path}, skipping.")
                    continue
                # Note: Assumes extract_keyframes_from_video is defined in utils.py
                extract_keyframes_from_video(original_video_path, video_keyframe_folder)

            # Execution: VLM Analysis
            res = video_processor.analyze_video(video_path, qa, max_frames=args.max_frames)

            # Construct Temporal Log
            video_content = split_points(res)
            video_content_str = derive_video_content_str(video_content)
            original_qa = question_fil(qa)

            # Re-assemble prompt for downstream Reasoner (e.g., A-MoE)
            qa_w_content = (
                "Please assume the role of an agent. The video represents your egocentric observations from the past to the present. "
                f"Video content: \n<\n{video_content_str}\n>\n"
                f"Question: {original_qa}"
            )

            # Atomic State Update
            qa_df.loc[idx, 'question'] = qa_w_content
            qa_df.loc[idx, 'processed'] = True

            # Real-time state persistence
            qa_df.to_json(save_path, orient='records', indent=4)
            
            elapsed_time = time.time() - start_time
            tqdm.write(f"✅ [Video ID: {video_name}] Log extracted successfully in {elapsed_time:.2f} seconds.")

        log.info(f"🎉 Completed log extraction for {data_path}\n")

if __name__ == '__main__':
    main()