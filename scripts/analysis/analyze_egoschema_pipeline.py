# -*- coding: utf-8 -*-
"""
High-Fidelity Visual Log Extraction for EgoSchema via 72B VLM
This script performs intelligent keyframe extraction using ORB feature matching 
to eliminate redundancy. It then utilizes a large-scale Vision-Language Model 
(e.g., Qwen2.5-VL-72B) to perform dual-frame contrastive analysis, generating 
rich textual semantic logs capturing fine-grained kinematics, force, and state changes.
"""

import argparse
import gc
import json
import logging
import os
import pickle
import re
import sys
import warnings

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm

# Qwen2.5-VL Specific Libraries
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("❌ Critical Error: Missing required libraries for Qwen2.5-VL.")
    print("Please install them using: pip install qwen-vl-utils")
    sys.exit(1)

# ==========================================
# Global Setup & Logging
# ==========================================
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_egoschema_logs.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# Phase 1: ORB Keyframe Extraction Protocol
# ==========================================
def calculate_overlap_ratio(corners1, corners2, width, height):
    """Calculates the bounding box overlap ratio to detect significant camera motion."""
    poly1 = np.array([c[0] for c in corners1], dtype=np.float32)
    poly2 = np.array([c[0] for c in corners2], dtype=np.float32)
    
    img1 = np.zeros((height, width), dtype=np.uint8)
    img2 = np.zeros((height, width), dtype=np.uint8)
    
    cv2.fillPoly(img1, np.int32([poly1]), 1)
    cv2.fillPoly(img2, np.int32([poly2]), 1)
    
    intersection = cv2.bitwise_and(img1, img2)
    overlap_area = np.sum(intersection)
    total_area = width * height
    return overlap_area / total_area

def constrained_sampling(lst, m):
    """Uniformly samples 'm' elements from a list, strictly preserving the first and last elements."""
    n = len(lst)
    if n <= m: return lst
    x = lst[0]
    y = lst[-1]
    if m <= 2: return [x, y][:m]
    
    num_samples = m - 2
    step = (n - 2) / num_samples
    sampled_indices = [int(round(i * step)) + 1 for i in range(num_samples)]
    sampled_indices = [min(max(idx, 1), n-2) for idx in sampled_indices]
    sampled_elements = [lst[idx] for idx in sampled_indices]
    return [x] + sampled_elements + [y]

def extract_keyframes_indices(frames, threshold=0.5):
    """Identifies keyframes based on ORB feature matching and Homography transformation."""
    keyframe_indices = [0]
    if len(frames) == 0: return []
    
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for t in range(1, len(frames)):
        curr_frame = frames[t]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        is_keyframe = False
        
        if prev_des is None or curr_des is None or len(prev_kp) < 2 or len(curr_kp) < 2:
            is_keyframe = True
        else:
            try:
                matches = bf.match(prev_des, curr_des)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:
                    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = prev_gray.shape
                        corners_prev = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        corners_curr_in_prev = cv2.perspectiveTransform(corners_prev, M)
                        overlap = calculate_overlap_ratio(corners_prev, corners_curr_in_prev, w, h)

                        if overlap < threshold:
                            is_keyframe = True
                    else:
                        is_keyframe = True 
                else:
                    is_keyframe = True 
            except Exception:
                is_keyframe = True

        if is_keyframe:
            keyframe_indices.append(t)
            prev_frame, prev_gray, prev_kp, prev_des = curr_frame, curr_gray, curr_kp, curr_des

    # Ensure the final frame is always included
    if keyframe_indices[-1] != len(frames) - 1:
        keyframe_indices.append(len(frames) - 1)
        
    return keyframe_indices

def get_smart_keyframes(video_path, max_frame=24, min_frame=2):
    """Extracts, filters, and bounds keyframes from a raw video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps)
    if interval <= 0: interval = 1
    
    downsampled_frames = []
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Sample at 1 FPS or force include the last frame
        if current_idx % interval == 0 or current_idx == frame_count - 1:
            downsampled_frames.append(frame)
        current_idx += 1
    cap.release()

    if not downsampled_frames: return []

    keyframe_indices = extract_keyframes_indices(downsampled_frames, threshold=0.5)

    # Bound the number of frames to prevent VLM OOM
    if len(keyframe_indices) < min_frame:
        full_indices = list(range(len(downsampled_frames)))
        keyframe_indices = constrained_sampling(full_indices, min_frame)
    elif len(keyframe_indices) > max_frame:
        keyframe_indices = constrained_sampling(keyframe_indices, max_frame)

    final_pil_images = []
    for idx in keyframe_indices:
        frame_bgr = downsampled_frames[idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        final_pil_images.append(Image.fromarray(frame_rgb))
        
    return final_pil_images

# ==========================================
# Phase 2: Qwen2.5-VL Perception Engine
# ==========================================
class QwenAnalyzer:
    """Manages the instantiation and querying of the VLM for frame-by-frame analysis."""
    
    def __init__(self, model_path: str):
        logger.info(f"🧠 Loading Vision-Language Model from: {model_path}...")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logger.info("✅ VLM loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load VLM: {e}")
            sys.exit(1)

    def clean_question(self, text: str) -> str:
        """Strips multiple-choice options, isolating the core question text."""
        if not text: return ""
        idx = text.find("Question:")
        if idx != -1: text = text[idx + len("Question:"):].strip()
        pattern = r"(?:\n\s*(?:Choices|Options|A\.|[A-H]\.)|(?<=[\?\.])\s*$)"
        split_text = re.split(pattern, text, flags=re.IGNORECASE)
        return split_text[0].strip()

    def analyze_frame_step(self, frame_curr, frame_prev, question: str, max_new_tokens: int = 384) -> str:
        """
        Executes dual-frame contrastive analysis. 
        Enforces domain-specific terminology and kinetic force estimation.
        """
        def resize_img(img, target_max=560):
            w, h = img.size
            if max(w, h) > target_max:
                ratio = target_max / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                return img.resize(new_size, Image.Resampling.LANCZOS)
            return img

        frame_curr = resize_img(frame_curr)
        if frame_prev: frame_prev = resize_img(frame_prev)

        # Prompt Formulation (Terminology & Kinematics)
        if frame_prev is None:
            # Initial Frame: Establish Spatial & Material Context
            system_prompt = (
                f"Question: \"{question}\"\n"
                "You are an expert video analyst. Analyze this INITIAL frame.\n"
                "Output STRICTLY in this format:\n"
                "1. Context & Materials: [Identify the setting (e.g., Construction Site, Lab). Identify materials precisely (e.g., use 'Mortar/Cement' instead of 'Mud' if in construction; 'Reagent' instead of 'Liquid' if in lab).]\n"
                "2. Hands & Tools: [Describe visible hands, gloves, and specific tools (e.g., 'Trowel' instead of 'Shovel', 'Pipette' instead of 'Tube').]\n"
                "3. q-Relevance: [Does the scene setup match the question's context?]"
            )
            image_inputs = [frame_curr]
        else:
            # Subsequent Frames: Sequential Differential Scan
            system_prompt = (
                f"Question: \"{question}\"\n"
                "You are an expert video forensic analyst. Compare the PREVIOUS frame (Image 1) with the CURRENT frame (Image 2).\n"
                "Focus on the HAND-OBJECT INTERACTION dynamics and force.\n\n"
                "Output STRICTLY in this format:\n"
                "1. Fine-grained Action: Use the format '[Hand] + [Verb] + [Object] + [Tool] + [Manner]'.\n"
                "   - **CRITICAL**: Describe the FORCE and VELOCITY. (e.g., use 'Slams'/'Throws' instead of 'Puts' if forceful; 'Scrubbing' instead of 'Wiping' if vigorous).\n"
                "   - **CRITICAL**: Use domain-specific terminology if applicable (e.g., 'Mortar', 'Solder', 'Whisk', 'Knead').\n"
                "   - Example: 'Right hand forcefully slams the lump of wet mortar into the wooden mold to remove air bubbles.'\n"
                "2. State Change: [Pre-state -> Post-state] (e.g., 'Mold was empty -> Mold is filled and overflowing').\n"
                "3. q-Relevance: [Link to Answer] How does this specific action (and its intensity) help distinguish the correct answer from distractors?"
            )
            image_inputs = [frame_prev, frame_curr]

        # VLM Inference execution
        messages = [{
            "role": "user", 
            "content": [{"type": "image", "image": img} for img in image_inputs] + [{"type": "text", "text": system_prompt}]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs_proc, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], 
            images=image_inputs_proc, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False 
            )
        
        output_text = self.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]
        
        return output_text

# ==========================================
# Main Orchestration
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract High-Fidelity Textual Logs via 72B VLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5-VL model (e.g., 72B-Int4)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing raw video files")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the EgoSchema metadata JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated JSONL logs")
    parser.add_argument("--cache_dir", type=str, default="./frames_cache", help="Directory to cache extracted ORB frames")
    parser.add_argument("--max_frames", type=int, default=24, help="Maximum number of frames to process per video")
    parser.add_argument("--max_new_tokens", type=int, default=384, help="Max tokens for the VLM generation step")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        logger.error(f"Input JSON not found at {args.input_json}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.input_json, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Resumption Protocol
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    processed_ids.add(json.loads(line)['id'])
                except: 
                    pass
    
    logger.info(f"📊 Progress: {len(processed_ids)} / {len(questions)} videos processed.")
    
    to_process = [item for item in questions if item['q_uid'] not in processed_ids]
    if not to_process:
        logger.info("🎉 All tasks completed!")
        sys.exit(0)

    analyzer = QwenAnalyzer(args.model_path)

    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        for item in tqdm(to_process, desc="VLM Log Extraction"):
            q_uid = item['q_uid']
            
            # Locate video file
            video_filename = item.get('video', f"{q_uid}.mp4")
            video_path = os.path.join(args.video_dir, video_filename)
            if not os.path.exists(video_path): 
                video_path = os.path.join(args.video_dir, f"{q_uid}.mp4")
            
            if not os.path.exists(video_path):
                tqdm.write(f"⚠️ Video not found: {video_path}. Skipping.")
                continue

            try:
                # Cache-aware Frame Extraction (ORB)
                cache_path = os.path.join(args.cache_dir, f"{q_uid}.pkl")
                frames = []
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        try: 
                            frames = pickle.load(f)
                        except Exception: 
                            pass
                
                if not frames:
                    frames = get_smart_keyframes(video_path, max_frame=args.max_frames)
                    if frames:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(frames, f)
                
                if not frames:
                    tqdm.write(f"⚠️ Frame extraction failed for {q_uid}. Skipping.")
                    continue

                # Sequential Semantic Extraction
                core_q = analyzer.clean_question(item['question'])
                analysis_results = []
                
                # Analyze Frame 0 (Initial Context)
                res0 = analyzer.analyze_frame_step(frames[0], None, core_q, max_new_tokens=args.max_new_tokens)
                analysis_results.append(f"[Frame 0] {res0}")
                
                # Analyze Frames 1 to N (Differential Kinematics)
                for i in range(1, len(frames)):
                    res = analyzer.analyze_frame_step(frames[i], frames[i-1], core_q, max_new_tokens=args.max_new_tokens)
                    analysis_results.append(f"[Frame {i}] {res}")

                # Compile final result package
                full_vlm_analysis = "\n\n".join(analysis_results)
                
                result = {
                    "id": q_uid,
                    "question": item['question'],
                    "answer": item.get('answer_label'),
                    "option_a": item.get('option 0'),
                    "option_b": item.get('option 1'),
                    "option_c": item.get('option 2'),
                    "option_d": item.get('option 3'),
                    "option_e": item.get('option 4'),
                    "vlm_analysis": full_vlm_analysis, 
                    "num_frames": len(frames)
                }
                
                # Atomic append
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
                # Aggressive memory management
                del frames
                del analysis_results
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"❌ Processing failed for {q_uid}: {e}")
                torch.cuda.empty_cache()

    logger.info("✅ Log extraction pipeline terminated successfully.")

if __name__ == "__main__":
    main()