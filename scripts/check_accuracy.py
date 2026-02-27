# -*- coding: utf-8 -*-
"""
Metrics Calculation Utility for A-MoE and Baseline Evaluations
This script parses the generated JSONL result files and computes 
the final reasoning accuracy and format compliance rate.
"""

import argparse
import json
import os
import sys

# ANSI Escape Codes for Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def calculate_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"{Colors.FAIL}❌ Error: File not found at {file_path}{Colors.ENDC}")
        sys.exit(1)

    total_count = 0
    correct_count = 0
    format_ok_count = 0

    print(f"{Colors.BLUE}📂 Analyzing results from: {file_path}{Colors.ENDC}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                total_count += 1
                
                # 1. Check Accuracy
                # Strictly check boolean True to avoid truthy edge cases
                if item.get("is_correct") is True:
                    correct_count += 1
                
                # 2. Check Format Compliance
                # Dynamically adapt to different key names across evaluation scripts
                pred = item.get("pred") or item.get("pred_letter") or ""
                is_format_ok = item.get("format_ok", False)
                
                if str(pred).upper() in ["A", "B", "C", "D", "E"] or is_format_ok:
                    format_ok_count += 1
                    
            except json.JSONDecodeError:
                print(f"{Colors.WARNING}⚠️ Warning: JSON parse error on line {line_num}. Skipping.{Colors.ENDC}")

    # ================= Generate Report =================
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        format_rate = (format_ok_count / total_count) * 100
        
        # Color-code the final accuracy based on thresholds (e.g., 50% for complex reasoning)
        if accuracy >= 60.0:
            acc_color = Colors.GREEN  # Excellent (e.g., A-MoE level)
        elif accuracy >= 40.0:
            acc_color = Colors.BLUE   # Good (e.g., SFT level)
        else:
            acc_color = Colors.WARNING # Baseline level
            
        print(f"{Colors.BOLD}{'=' * 50}{Colors.ENDC}")
        print(f"{Colors.BOLD}🎯 EVALUATION METRICS REPORT{Colors.ENDC}")
        print(f"{Colors.BOLD}{'=' * 50}{Colors.ENDC}")
        print(f"📝 Total Samples Processed : {total_count}")
        print(f"✅ Correct Predictions     : {correct_count}")
        print(f"❌ Incorrect Predictions   : {total_count - correct_count}")
        print(f"{'-' * 50}")
        print(f"{Colors.BOLD}📊 Final Accuracy          : {acc_color}{accuracy:.2f}%{Colors.ENDC}")
        print(f"{Colors.BOLD}🧮 Format Compliance Rate  : {format_rate:.2f}%{Colors.ENDC}")
        print(f"{Colors.BOLD}{'=' * 50}{Colors.ENDC}\n")
    else:
        print(f"{Colors.WARNING}⚠️ No valid JSON data found in the file.{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from JSONL result files.")
    parser.add_argument(
        "--result_file", 
        type=str, 
        required=True, 
        help="Path to the JSONL file containing evaluation results (e.g., RL-egoschema.jsonl)"
    )
    args = parser.parse_args()

    calculate_metrics(args.result_file)

if __name__ == "__main__":
    main()