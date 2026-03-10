"""
Launcher script. Pushes train.py and prepare.py to Kaggle to be executed on a T4 GPU.
Polls for completion and downloads the execution log.
"""
import os
import json
import time
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()

def main():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        username = api.get_config_value("username")
    except Exception as e:
        print("Failed to authenticate Kaggle API. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in .env")
        print(e)
        return

    run_id = f"{username}/autoresearch-pinn-lle"
    submit_dir = "kaggle_submit"
    os.makedirs(submit_dir, exist_ok=True)
    
    # Copy scripts to submission folder (they will sit next to each other on Kaggle)
    shutil.copy("train.py", os.path.join(submit_dir, "train.py"))
    shutil.copy("prepare.py", os.path.join(submit_dir, "prepare.py"))
    
    # Metadata for Kaggle Kernel explicitly requesting GPU T4
    metadata = {
        "id": run_id,
        "title": "autoresearch-pinn-lle",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "accelerator": "GPU", # Explicit Kaggle T4 mapping
        "enable_internet": "true",
        "dataset_sources": ["technolight/matlab-conditions"],
        "competition_sources":[],
        "kernel_sources": [],
        "model_sources":[]
    }
    
    with open(os.path.join(submit_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Pushing job to Kaggle ({run_id})...")
    subprocess.run(["kaggle", "kernels", "push", "-p", submit_dir], check=True)
    
    print("Waiting for Kaggle GPU execution to finish (this takes up to 15 mins)...")
    while True:
        time.sleep(30)
        res = subprocess.run(["kaggle", "kernels", "status", run_id], capture_output=True, text=True)
        output = res.stdout.strip()
        
        if "complete" in output or "error" in output or "cancel" in output:
            print(f"\nFinal status: {output}")
            break
        elif "running" in output or "queued" in output:
            print(f"Status: {output}")
        else:
            print(f"Unknown status: {output}")
            break
            
    # Download logs
    out_dir = "kaggle_output"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", run_id, "-p", out_dir], check=False)
    
    log_files = [f for f in os.listdir(out_dir) if f.endswith(".log")]
    if log_files:
        log_path = os.path.join(out_dir, log_files[0])
        with open(log_path, "r") as f:
            print("\n" + "="*40)
            print("KAGGLE RUN OUTPUT")
            print("="*40)
            print(f.read())
    else:
        print("\nNo .log found in output. The kernel might have crashed early or syntax error.")

if __name__ == "__main__":
    main()