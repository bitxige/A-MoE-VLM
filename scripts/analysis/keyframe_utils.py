# -*- coding: utf-8 -*-
"""
ORB-based Keyframe Extraction Utility for A-MoE Framework
This module provides video downsampling and intelligent keyframe extraction 
based on Oriented FAST and Rotated BRIEF (ORB) feature matching and homography calculation.
It filters out redundant visual information, retaining only significant dynamic transitions.
"""

import os
import cv2
import numpy as np


def downsample_video(video_path):
    """
    Downsample the video to 1 fps while strictly retaining the first and last frames.
    
    Args:
        video_path (str): Path to the source video file.
        
    Returns:
        list: A list of downsampled frame arrays (numpy.ndarray).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Warning: Failed to open video file at {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    downsampled_frames = []
    
    interval = int(fps)
    if interval == 0:
        interval = 1
        
    frame_indices_to_keep = list(range(0, frame_count, interval))
    
    # Ensure the final frame is always included
    if (frame_count - 1) not in frame_indices_to_keep:
        frame_indices_to_keep.append(frame_count - 1)
        
    success, frame = cap.read()
    index = 0
    while success:
        if index in frame_indices_to_keep:
            downsampled_frames.append(frame)
        success, frame = cap.read()
        index += 1
        
    cap.release()
    return downsampled_frames


def extract_keyframes_from_video(video_path, output_dir, max_frame=32, min_frame=4):
    """
    End-to-end pipeline to extract intelligent keyframes from a video and save them.

    Args:
        video_path (str): Path of the source video file.
        output_dir (str): Directory to save the output keyframe video/frames.
        max_frame (int): Maximum allowable number of keyframes to prevent VRAM overflow.
        min_frame (int): Minimum required number of keyframes for valid contextual reasoning.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Baseline Downsampling (1 FPS)
    frames = downsample_video(video_path)
    if not frames or len(frames) == 0:
        print(f"⚠️ Warning: Video {video_path} downsampling failed or is empty.")
        return
        
    frame_count = len(frames)

    # Step 2: ORB-based intelligent extraction
    overlap_threshold = 0.5
    keyframe_indices = extract_keyframes(frames, overlap_threshold)
    
    print(f"📸 [Video: {os.path.basename(video_path)}]")
    print(f"   -> Extracted Keyframe Indices: {keyframe_indices}")
    print(f"   -> Total Keyframes Selected: {len(keyframe_indices)}")

    # Step 3: Constraint enforcing (Truncation or Padding)
    if len(keyframe_indices) < min_frame:
        full_indices = list(range(frame_count))
        keyframe_indices = constrained_sampling(full_indices, min_frame)
    elif len(keyframe_indices) > max_frame:
        keyframe_indices = constrained_sampling(keyframe_indices, max_frame)

    # Step 4: Output Rendering
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    
    frame_height, frame_width, _ = frames[0].shape
    fps_out = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (frame_width, frame_height))
    
    for idx in keyframe_indices:
        out.write(frames[idx])
        
    out.release()
    print(f"   -> Successfully saved unified keyframe compilation to: {output_path}")


def extract_keyframes(frames, overlap_threshold):
    """
    Extract keyframes dynamically based on ORB feature matching and homography 
    coincidence ratios between consecutive chronological frames.

    Args:
        frames (list): Sequential list of video frames (numpy arrays).
        overlap_threshold (float): Spatial overlap threshold (0 to 1) indicating scene redundancy.

    Returns:
        list: Filtered indices of frames designated as keyframes.
    """
    keyframe_indices = [0]  # First frame is strictly retained
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB feature detector and Hamming-distance brute-force matcher
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for t in range(1, len(frames)):
        curr_frame = frames[t]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        # Trigger fallback if descriptor computation fails
        if prev_des is None or curr_des is None:
            keyframe_indices.append(t)
            prev_frame, prev_gray, prev_kp, prev_des = curr_frame, curr_gray, curr_kp, curr_des
            continue

        # Execute feature matching
        matches = bf.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Homography estimation requires sufficient match points
        if len(matches) > 10:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # RANSAC-based transformation matrix estimation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = prev_gray.shape
                corners_prev = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                corners_curr_in_prev = cv2.perspectiveTransform(corners_prev, M)

                overlap_ratio = calculate_overlap_ratio(corners_prev, corners_curr_in_prev, w, h)

                # Register as keyframe if displacement exceeds threshold
                if overlap_ratio < overlap_threshold:
                    keyframe_indices.append(t)
                    prev_frame, prev_gray, prev_kp, prev_des = curr_frame, curr_gray, curr_kp, curr_des
            else:
                # Transformation collapse, force keyframe
                keyframe_indices.append(t)
                prev_frame, prev_gray, prev_kp, prev_des = curr_frame, curr_gray, curr_kp, curr_des
        else:
            # Extreme visual disparity (too few matches), force keyframe
            keyframe_indices.append(t)
            prev_frame, prev_gray, prev_kp, prev_des = curr_frame, curr_gray, curr_kp, curr_des

    # Ensure the terminal frame is strictly retained for action completeness
    if keyframe_indices[-1] != len(frames) - 1:
        keyframe_indices.append(len(frames) - 1)
        
    return keyframe_indices


def calculate_overlap_ratio(corners1, corners2, width, height):
    """
    Calculate the intersection-over-union equivalent overlap ratio between two projected polygons.

    Args:
        corners1 (np.ndarray): Corner coordinates of the reference polygon.
        corners2 (np.ndarray): Projected corner coordinates of the target polygon.
        width (int): Frame width.
        height (int): Frame height.

    Returns:
        float: The ratio of the intersection area to the total frame area.
    """
    poly1 = np.array([c[0] for c in corners1], dtype=np.float32)
    poly2 = np.array([c[0] for c in corners2], dtype=np.float32)

    poly1_int = np.int32([poly1])
    poly2_int = np.int32([poly2])

    # Rasterize polygons into binary masks
    img1 = np.zeros((height, width), dtype=np.uint8)
    img2 = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(img1, poly1_int, 1)
    cv2.fillPoly(img2, poly2_int, 1)

    intersection = cv2.bitwise_and(img1, img2)
    overlap_area = np.sum(intersection)
    total_area = width * height
    
    return overlap_area / total_area


def constrained_sampling(lst, m):
    """
    Adjusts the sampled frame list to strictly adhere to boundary length constraints 
    via uniform intermediate interpolation.

    Args:
        lst (list): Source list of frame indices.
        m (int): Target constraint length (either max or min boundary).

    Returns:
        list: Normalized list of frame indices.
    """
    n = len(lst)
    if n <= m:
        return lst
        
    x = lst[0]
    y = lst[-1]

    # Uniformly resample intermediate points
    num_samples = m - 2
    step = (n - 2) / num_samples
    sampled_indices = [int(round(i * step)) + 1 for i in range(num_samples)]
    sampled_elements = [lst[idx] for idx in sampled_indices]

    if sampled_elements and sampled_elements[-1] == y:
        print("⚠️ Constraint Warning: Terminal element overlap detected during subsampling.")

    return [x] + sampled_elements + [y]