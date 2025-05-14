#!/usr/bin/env python
# coding: utf-8

# # Quick Model Architecture Check for microWakeWord
#
# This notebook performs a very quick training run with minimal data to check if
# the model architecture is fundamentally working and can learn something.
# It is NOT intended for training a usable wake word model.
# Assumes Python 3.11 and necessary dependencies are installed.

import os
import sys
import platform
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import scipy.io.wavfile
import numpy as np
import yaml
import json

# Conditional import for display
try:
    from IPython.display import Audio, display
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass

print(f"Python version: {sys.version}")
print(f"Running in directory: {os.getcwd()}")

# --- Configuration & Data Paths ---
BASE_DATA_DIR_QUICK_CHECK = Path("./mww_quick_check_data")
BASE_DATA_DIR_QUICK_CHECK.mkdir(parents=True, exist_ok=True)
print(f"Using data directory for this quick check: {BASE_DATA_DIR_QUICK_CHECK.resolve().absolute()}")

PIPER_REPO_DIR_QC = BASE_DATA_DIR_QUICK_CHECK / "piper-sample-generator"
PIPER_MODEL_FILENAME_QC = "en_US-libritts_r-medium.pt"
PIPER_MODEL_URL_QC = f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_FILENAME_QC}"
PIPER_MODEL_DIR_QC = PIPER_REPO_DIR_QC / "models"
PIPER_MODEL_FILE_QC = PIPER_MODEL_DIR_QC / PIPER_MODEL_FILENAME_QC

GENERATED_SAMPLES_DIR_QC = BASE_DATA_DIR_QUICK_CHECK / "generated_samples_qc"
NEGATIVE_DATASETS_DIR_QC = BASE_DATA_DIR_QUICK_CHECK / "negative_datasets_qc"
GENERATED_FEATURES_DIR_QC = BASE_DATA_DIR_QUICK_CHECK / "generated_features_qc" # Simplified name
TRAINED_MODELS_DIR_QC = BASE_DATA_DIR_QUICK_CHECK / "trained_models_qc/wakeword"

# --- Helper Functions (Simplified) ---
def run_command_qc(command_list, description, cwd=None):
    print(f"Executing: {description} -> {' '.join(command_list)}")
    try:
        subprocess.run(command_list, check=True, capture_output=True, text=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed {description}.\nCmd: {e.cmd}\nCode: {e.returncode}\nOutput: {e.output}\nStderr: {e.stderr}")
        return False

def download_file_qc(url, output_path, description="file"):
    output_path = Path(output_path)
    if output_path.exists(): print(f"{description} exists. Skipping."); return True
    print(f"Downloading {description} from {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(output_path, "wb") as f: shutil.copyfileobj(response.raw, f)
        return True
    except Exception as e: print(f"Error downloading {url}: {e}"); return False

def extract_zip_qc(zip_path, extract_to, expected_content_name=None):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    if expected_content_name and (extract_to / expected_content_name).is_dir():
        print(f"'{expected_content_name}' seems extracted in {extract_to}. Skipping.")
        return True
    if not zip_path.exists(): print(f"ZIP not found: {zip_path}"); return False
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_to)
        return True
    except Exception as e: print(f"Error extracting {zip_path}: {e}"); return False

# --- Data Preparation (Minimal) ---
print("\n--- Preparing Minimal Dependencies and Data for Quick Check ---")
# 1. Piper Sample Generator
PIPER_REPO_DIR_QC.mkdir(parents=True, exist_ok=True)
if not (PIPER_REPO_DIR_QC / ".git").is_dir():
    run_command_qc(["git", "clone", "https://github.com/rhasspy/piper-sample-generator.git", str(PIPER_REPO_DIR_QC)], "Cloning Piper")
PIPER_MODEL_DIR_QC.mkdir(parents=True, exist_ok=True)
download_file_qc(PIPER_MODEL_URL_QC, PIPER_MODEL_FILE_QC, "Piper TTS Model")
if str(PIPER_REPO_DIR_QC) not in sys.path: sys.path.append(str(PIPER_REPO_DIR_QC))

# 2. Negative Data (Just one small set, pre-generated features)
NEGATIVE_DATASETS_DIR_QC.mkdir(parents=True, exist_ok=True)
neg_fname_qc = 'no_speech.zip'
neg_zip_path_qc = NEGATIVE_DATASETS_DIR_QC / neg_fname_qc
if download_file_qc(f"https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/{neg_fname_qc}", neg_zip_path_qc, "Negative data (no_speech features)"):
    extract_zip_qc(neg_zip_path_qc, NEGATIVE_DATASETS_DIR_QC, "no_speech")
print("--- Minimal Data Preparation Finished ---")

# Generate a few positive samples
target_word_qc = 'test_word'
print(f"\n--- Generating a few samples for '{target_word_qc}' ---")
GENERATED_SAMPLES_DIR_QC.mkdir(parents=True, exist_ok=True)
piper_script_qc = PIPER_REPO_DIR_QC / "generate_samples.py"
cmd_gen_samples_qc = [sys.executable, str(piper_script_qc), target_word_qc,
                      "--max-samples", "50", "--batch-size", "10",
                      "--output-dir", str(GENERATED_SAMPLES_DIR_QC)]
if piper_script_qc.exists():
    run_command_qc(cmd_gen_samples_qc, "Generating few positive samples")
else:
    print(f"ERROR: Piper script not found at {piper_script_qc}")

# Feature Generation (No explicit augmentation for quick check)
print("\n--- Generating Features (No Augmentation) ---")
# Ensure microwakeword modules are importable (assuming installed in environment)
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

GENERATED_FEATURES_DIR_QC.mkdir(parents=True, exist_ok=True)
# Positive features
positive_clips_qc = Clips(input_directory=str(GENERATED_SAMPLES_DIR_QC), file_pattern='*.wav', remove_silence=True) # Apply VAD
positive_spectrograms_qc = SpectrogramGeneration(clips=positive_clips_qc, augmenter=None, slide_frames=1, step_ms=10) # No augmentation, no sliding for simplicity
out_dir_pos_qc = GENERATED_FEATURES_DIR_QC / "training" / "positive_mmap"
out_dir_pos_qc.mkdir(parents=True, exist_ok=True)
if GENERATED_SAMPLES_DIR_QC.is_dir() and any(GENERATED_SAMPLES_DIR_QC.glob("*.wav")):
    print("Generating positive features...")
    RaggedMmap.from_generator(
        out_dir=str(out_dir_pos_qc),
        sample_generator=positive_spectrograms_qc.spectrogram_generator(split="train", repeat=1), # Using 'train' split from Clips
        batch_size=25, verbose=True
    )
else:
    print("No positive samples found to generate features.")

# Negative features (already pre-generated, just need to point to them)
# The no_speech.zip already contains mmap folders for training, validation, testing.
# We'll just use its training split.
print("Negative features are pre-generated in 'no_speech' dataset.")


# Training Configuration (Minimal)
print("\n--- Preparing Minimal Training Configuration ---")
config_yaml_qc = {
    "window_step_ms": 10,
    "train_dir": str(TRAINED_MODELS_DIR_QC),
    "features": [
        {"features_dir": str(GENERATED_FEATURES_DIR_QC), "sampling_weight": 1.0, "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
        {"features_dir": str(NEGATIVE_DATASETS_DIR_QC / "no_speech"), "sampling_weight": 1.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
    ],
    "training_steps": [200], # Very few steps
    "positive_class_weight": [1.0],
    "negative_class_weight": [1.0], # Balanced for quick check
    "learning_rates": [0.001],
    "batch_size": 32, # Smaller batch size
    "time_mask_max_size": [0], "time_mask_count": [0], # No SpecAugment
    "freq_mask_max_size": [0], "freq_mask_count": [0],
    "eval_step_interval": 50, # Evaluate more frequently
    "clip_duration_ms": 1000, # Shorter clip duration for faster processing
    "target_minimization": 1.0, # Don't be strict
    "minimization_metric": None,
    "maximization_metric": "accuracy" # Just check if accuracy improves
}
TRAINED_MODELS_DIR_QC.mkdir(parents=True, exist_ok=True)
training_params_yaml_path_qc = BASE_DATA_DIR_QUICK_CHECK / "quick_check_training_parameters.yaml"
with open(training_params_yaml_path_qc, "w") as f_yaml_qc: yaml.dump(config_yaml_qc, f_yaml_qc)
print(f"Quick check training parameters saved to {training_params_yaml_path_qc}")

# Model Training (Quick)
print("\n--- Starting Quick Model Training ---")
cmd_train_qc = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(training_params_yaml_path_qc),
    "--train", "1", "--restore_checkpoint", "0", # No restore for quick check
    "--test_tflite_streaming_quantized", "0", # No extensive testing
    "mixednet", # Default model, can be changed
    "--pointwise_filters", "32,32", "--repeat_in_block", "1,1", # Smaller model
    "--mixconv_kernel_sizes", "[3],[5]", "--first_conv_filters", "16",
    "--first_conv_kernel_size", "3", "--stride", "2"
]
run_command_qc(cmd_train_qc, "Quick model training")

print("\n--- Quick Model Check Finished ---")
output_tflite_path_qc = TRAINED_MODELS_DIR_QC / "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
if output_tflite_path_qc.exists():
    print(f"Quick check model (if generated by test phase) might be at: {output_tflite_path_qc.resolve().absolute()}")
else:
    print(f"Quick check training completed. Check logs in {TRAINED_MODELS_DIR_QC} for model status.")
    print("Note: Full TFLite conversion might be skipped if --test_tflite_streaming_quantized is 0.")