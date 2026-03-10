"""
Launcher script. Pushes train.py and prepare.py to Kaggle to be executed on a T4 GPU.
Polls for completion and downloads the execution log.
"""
import os
import json
import time
import shutil
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

def main():
    # Проверяем, что заданы нужные переменные (включая новую KAGGLE_API_TOKEN)
    username = os.environ.get("KAGGLE_USERNAME")
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    
    if not username or not api_token:
        print("[ERROR] В файле .env должны быть заданы KAGGLE_USERNAME и KAGGLE_API_TOKEN (формата KGAT_...)")
        return

    # Импортируем KaggleApi (версии >= 1.8.0 автоматически подхватят KAGGLE_API_TOKEN из os.environ)
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    try:
        api = KaggleApi()
        api.authenticate()
        print(f"[OK] Authenticated via KAGGLE_API_TOKEN as {username}")
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        return

    run_id = f"{username}/autoresearch-pinn-lle"
    submit_dir = "kaggle_submit"
    os.makedirs(submit_dir, exist_ok=True)
    
    # Копируем скрипты
    shutil.copy("train.py", os.path.join(submit_dir, "train.py"))
    shutil.copy("prepare.py", os.path.join(submit_dir, "prepare.py"))
    
    # Метаданные (с булевыми значениями True/False)
    metadata = {
        "id": run_id,
        "title": "autoresearch-pinn-lle",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "GPU", # Запрос T4 GPU
        "enable_internet": True,
        "dataset_sources":["technolight/matlab-conditions"],
        "competition_sources": [],
        "kernel_sources":[],
        "model_sources":[]
    }
    
    with open(os.path.join(submit_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[*] Pushing job to Kaggle ({run_id})...")
    try:
        api.kernels_push(folder=submit_dir)
        print("[OK] Kernel pushed successfully!")
    except Exception as e:
        print(f"[ERROR] Kernel push failed: {e}")
        return
        
    print("[*] Waiting for Kaggle GPU execution to finish (this takes up to 15 mins)...")
    last_status = None
    
    while True:
        time.sleep(30)
        try:
            # Получаем статус напрямую через объект API
            res = api.kernels_get_status(run_id)
            status = str(res).lower()
            
            if status != last_status:
                print(f"Status: {status}")
                last_status = status
            
            if "complete" in status:
                print(f"\n[OK] Final status: {status}")
                break
            elif "error" in status or "cancel" in status or "fail" in status:
                print(f"\n[ERROR] Final status: {status}")
                break
        except Exception as e:
            print(f"[WARN] Status check error (ignoring): {e}")
            
    # Скачивание логов
    out_dir = "kaggle_output"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Downloading output to {out_dir}...")
    try:
        api.kernels_output(run_id, out_dir)
    except Exception as e:
        print(f"[WARN] Error downloading output: {e}")
    
    # Чтение лога
    log_files =[f for f in os.listdir(out_dir) if f.endswith(".log")]
    if log_files:
        log_path = os.path.join(out_dir, log_files[0])
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            print("\n" + "="*40)
            print("KAGGLE RUN OUTPUT")
            print("="*40)
            print(f.read())
    else:
        print("\n[ERROR] No .log found in output. The kernel might have crashed early or had a syntax error.")

if __name__ == "__main__":
    main()