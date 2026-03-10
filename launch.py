"""
Launcher script. Pushes train.py and prepare.py to Kaggle to be executed on a T4 GPU.
Polls for completion, handles errors, and downloads the execution log.
"""
import os
import json
import time
import shutil
import traceback
from dotenv import load_dotenv

load_dotenv()

def main():
    username = os.environ.get("KAGGLE_USERNAME")
    api_token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    
    if not username or not api_token:
        print("[ERROR] KAGGLE_USERNAME and KAGGLE_API_TOKEN (or KAGGLE_KEY) must be set in .env")
        return

    # Initialize Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[INFO] Authenticated via Kaggle API as {username}")
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        return

    run_id = f"{username}/autoresearch-pinn-lle"
    submit_dir = "kaggle_submit"
    os.makedirs(submit_dir, exist_ok=True)
    
    # Copy necessary files
    shutil.copy("train.py", os.path.join(submit_dir, "train.py"))
    shutil.copy("prepare.py", os.path.join(submit_dir, "prepare.py"))
    
    # Metadata: EXPLICITLY requesting NVIDIA_TESLA_T4
    metadata = {
        "id": run_id,
        "title": "autoresearch-pinn-lle",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "NVIDIA_TESLA_T4",  # Force T4 instead of P100
        "enable_internet": True,
        "dataset_sources": ["technolight/matlab-conditions"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources":[]
    }
    
    with open(os.path.join(submit_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[INFO] Pushing job to Kaggle ({run_id})...")
    try:
        api.kernels_push(folder=submit_dir)
        print("[INFO] Kernel pushed successfully. Job is in queue.")
    except Exception as e:
        print(f"[ERROR] Kernel push failed. Exception:\n{traceback.format_exc()}")
        return
        
    print("[INFO] Waiting for Kaggle T4 execution to finish (can take 15-20 mins)...")
    last_status = None
    
    # Polling loop
    while True:
        time.sleep(30)
        try:
            res = api.kernels_get_status(run_id)
            status = str(res).strip().lower()
            
            if status != last_status:
                print(f"[INFO] Current Kaggle Status: {status}")
                last_status = status
            
            if "complete" in status:
                print(f"[INFO] Final status reached: {status}")
                break
            elif any(err in status for err in["error", "cancel", "fail", "timeout"]):
                print(f"[ERROR] Execution stopped with status: {status}")
                break
        except Exception as e:
            print(f"[WARN] Status check error (network glitch? Retrying...): {e}")
            
    # Download Logs
    out_dir = "kaggle_output"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Downloading output logs to {out_dir}...")
    try:
        api.kernels_output(run_id, out_dir)
    except Exception as e:
        print(f"[WARN] Error downloading output: {e}")
    
    # Parse and Print Log
    log_files =[f for f in os.listdir(out_dir) if f.endswith(".log")]
    if log_files:
        log_path = os.path.join(out_dir, log_files[0])
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                print("\n" + "="*50)
                print("KAGGLE RUN OUTPUT")
                print("="*50)
                print(f.read())
                print("="*50)
        except Exception as e:
            print(f"[ERROR] Failed to read downloaded log: {e}")
    else:
        print("\n[ERROR] No .log found in output. The kernel crashed early, OOMed, or had a syntax error.")

if __name__ == "__main__":
    main()