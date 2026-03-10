"""
Launcher script. Pushes the job to Kaggle using the official v2.0.0 CLI.
Polls for completion, handles queues, and downloads the execution log.
"""
import os
import json
import time
import subprocess
import re
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def run_cmd(cmd):
    """Helper to run Kaggle CLI commands with UTF-8 encoding safeguard"""
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" in env and "KAGGLE_KEY" not in env:
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
        
    # encoding='utf-8' prevents Windows charmap crash if Kaggle outputs special chars
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    return res.returncode, res.stdout, res.stderr

def main():
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        print("[ERROR] KAGGLE_USERNAME must be set in .env")
        return

    run_id = f"{username}/autoresearch-pinn-lle"
    submit_dir = "kaggle_submit"
    os.makedirs(submit_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Merge prepare.py and train.py
    # ---------------------------------------------------------
    print("[INFO] Merging prepare.py and train.py for Kaggle execution...")
    try:
        with open("prepare.py", "r", encoding="utf-8") as f:
            prepare_code = f.read()
        with open("train.py", "r", encoding="utf-8") as f:
            train_code = f.read()

        # Remove the import line from train.py
        train_code = re.sub(r'^from\s+prepare\s+import\s+.*$', '', train_code, flags=re.MULTILINE)

        merged_code = prepare_code + "\n\n# " + "="*30 + "\n# BEGIN TRAIN.PY\n# " + "="*30 + "\n\n" + train_code

        with open(os.path.join(submit_dir, "train.py"), "w", encoding="utf-8") as f:
            f.write(merged_code)
    except Exception as e:
        print(f"[ERROR] Failed to merge Python files: {e}")
        return

    # ---------------------------------------------------------
    # 2. Kernel Metadata
    # ---------------------------------------------------------
    metadata = {
        "id": run_id,
        "title": "autoresearch-pinn-lle",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": ["technolight/matlab-conditions"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    
    with open(os.path.join(submit_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # ---------------------------------------------------------
    # 3. Push to Kaggle with T4 and 21min timeout
    # ---------------------------------------------------------
    print(f"[INFO] Pushing job to Kaggle ({run_id}) with T4 and 21min timeout...")
    
    code, out, err = run_cmd(f'kaggle kernels push -p {submit_dir} --accelerator NvidiaTeslaT4 --timeout 1260')
    if code != 0:
        print(f"[ERROR] Kernel push failed. Code: {code}")
        print(f"STDOUT: {out.strip()}\nSTDERR: {err.strip()}")
        return
        
    print("[INFO] Kernel pushed successfully. Job is in queue.")
        
    # ---------------------------------------------------------
    # 4. Polling loop
    # ---------------------------------------------------------
    print("[INFO] Waiting for Kaggle execution to finish...")
    last_status = None
    consecutive_errors = 0
    
    while True:
        time.sleep(30)
        code, out, err = run_cmd(f'kaggle kernels status {run_id}')
        
        if code != 0:
            consecutive_errors += 1
            print(f"[WARN] Status check error {consecutive_errors}/5: {err.strip() or out.strip()}")
            if consecutive_errors >= 5:
                print("[ERROR] Too many network/CLI failures. Aborting poll.")
                break
            continue
            
        consecutive_errors = 0 
        
        status_raw = out.strip().lower()
        match = re.search(r'status "([^"]+)"', status_raw)
        status = match.group(1) if match else status_raw
        
        if status != last_status:
            print(f"[INFO] Current Kaggle Status: {status}")
            last_status = status
        
        if "complete" in status:
            print(f"\n[INFO] Final status reached: {status}")
            break
        elif any(s in status for s in ["error", "cancel", "fail", "timeout"]):
            print(f"\n[ERROR] Execution stopped with status: {status}")
            break
            
    # ---------------------------------------------------------
    # 5. Download Output
    # ---------------------------------------------------------
    out_dir = "kaggle_output"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Downloading output logs to {out_dir}...")
    
    code, out, err = run_cmd(f'kaggle kernels output {run_id} -p {out_dir} -o -q')
    if code != 0:
        print(f"[WARN] Error downloading output: {err.strip() or out.strip()}")
    
    log_files = [f for f in os.listdir(out_dir) if f.endswith(".log")]
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
        print("\n[ERROR] No .log found in output. The kernel crashed early, OOMed, or timed out without saving logs.")

if __name__ == "__main__":
    main()