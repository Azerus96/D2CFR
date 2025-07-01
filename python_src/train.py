import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess

# --- НАСТРОЙКИ ---
NUM_CPP_WORKERS = int(os.cpu_count() or 96) - 8 
NUM_COMPUTATION_THREADS = "8"
NUM_INFERENCE_WORKERS = 8 

os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

from .model import DuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer, InferenceQueue

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 100
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 5_000_000
BATCH_SIZE = 8192
SAVE_INTERVAL_SECONDS = 1800
MODEL_PATH = "d2cfr_model.pth"

INFERENCE_BATCH_SIZE = 1024

class InferenceWorker(threading.Thread):
    def __init__(self, model_provider, queue, device, worker_id):
        super().__init__(daemon=True)
        self.model_provider = model_provider
        self.queue = queue
        self.device = device
        self.worker_id = worker_id
        # --- ИЗМЕНЕНИЕ: Используем общий stop_event из model_provider ---
        self.stop_event = self.model_provider.stop_event

    def run(self):
        print(f"InferenceWorker-{self.worker_id} (ThreadID: {threading.get_ident()}) started.", flush=True)
        model = self.model_provider.get_model()
        
        while not self.stop_event.is_set():
            try:
                if self.model_provider.is_updated(self.worker_id):
                    model = self.model_provider.get_model()
                    self.model_provider.mark_updated(self.worker_id)
                
                requests = self.queue.pop_n(INFERENCE_BATCH_SIZE)
                
                if not requests:
                    # Если pop_n вернул пустой вектор, это может быть сигнал остановки
                    if self.stop_event.is_set():
                        break
                    continue
                
                self.process_batch(requests, model)

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error in InferenceWorker-{self.worker_id}: {e}", flush=True)
                    traceback.print_exc()

        print(f"InferenceWorker-{self.worker_id} (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def process_batch(self, requests, model):
        infosets = [req.infoset for req in requests]
        tensor = torch.tensor(infosets, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            results_tensor = model(tensor)
        
        results_list = results_tensor.cpu().numpy()

        for i, req in enumerate(requests):
            result = results_list[i][:req.num_actions].tolist()
            req.set_result(result)

class InferenceModelProvider:
    def __init__(self, model, device, num_workers):
        self.device = device
        self.num_workers = num_workers
        self.model_lock = threading.Lock()
        self.updated_flags = [True] * num_workers
        self.stop_event = threading.Event() # <-- Общий флаг остановки
        self.set_model(model)

    def set_model(self, model):
        with self.model_lock:
            self.model = torch.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
            self.updated_flags = [True] * self.num_workers
        print("New inference model is ready for all workers.", flush=True)

    def get_model(self):
        with self.model_lock:
            return self.model

    def is_updated(self, worker_id):
        return self.updated_flags[worker_id]

    def mark_updated(self, worker_id):
        self.updated_flags[worker_id] = False

def push_to_github(model_path, commit_message):
    try:
        print("Pushing progress to GitHub...", flush=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True)
        subprocess.run(['git', 'add', model_path], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Progress pushed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except RuntimeError as e:
            print(f"Could not load state_dict. Error: {e}. Starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    inference_queue = InferenceQueue()

    model_provider = InferenceModelProvider(model, device, NUM_INFERENCE_WORKERS)
    inference_workers = [
        InferenceWorker(model_provider, inference_queue, device, i)
        for i in range(NUM_INFERENCE_WORKERS)
    ]
    for worker in inference_workers:
        worker.start()

    solvers = [DeepMCCFR(ACTION_LIMIT, replay_buffer, inference_queue) for _ in range(NUM_CPP_WORKERS)]
    
    stop_event = threading.Event()
    total_samples_generated = 0
    git_thread = None

    try:
        with ThreadPoolExecutor(max_workers=NUM_CPP_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_CPP_WORKERS} long-running C++ worker tasks...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_save_time = time.time()
            last_report_time = time.time()
            last_buffer_size = 0
            loss = None

            while True:
                time.sleep(0.1)
                
                current_buffer_size = replay_buffer.get_count()
                now = time.time()
                
                if current_buffer_size >= last_buffer_size + BATCH_SIZE:
                    model.train()
                    infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                    
                    infosets = torch.from_numpy(infosets_np).to(device)
                    targets = torch.from_numpy(targets_np).to(device)

                    optimizer.zero_grad()
                    predictions = model(infosets)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    model_provider.set_model(model)
                    
                    last_buffer_size = current_buffer_size
                
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    samples_generated_interval = current_buffer_size - total_samples_generated
                    total_samples_generated = current_buffer_size
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Buffer: {current_buffer_size}/{REPLAY_BUFFER_CAPACITY}. Total generated: {total_samples_generated:,}", flush=True)
                    
                    if loss is not None:
                        print(f"Last training loss: {loss.item():.6f}", flush=True)

                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        if git_thread and git_thread.is_alive():
                            print("Previous Git push is still running. Skipping this save.", flush=True)
                        else:
                            if loss is not None:
                                print("\n--- Saving model and pushing to GitHub ---", flush=True)
                                torch.save(model.state_dict(), MODEL_PATH)
                                commit_message = f"Training checkpoint. Total samples: {total_samples_generated:,}. Loss: {loss.item():.6f}"
                                
                                git_thread = threading.Thread(target=push_to_github, args=(MODEL_PATH, commit_message))
                                git_thread.start()
                                
                                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping workers...")
        stop_event.set()
        
        # --- ИЗМЕНЕНИЕ: Правильная и надежная остановка ---
        model_provider.stop_event.set()
        inference_queue.stop() # Это разбудит все инференс-воркеры
        
        print("Waiting for C++ workers to finish...")
        for future in futures:
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"C++ worker finished with an exception: {e}")

        print("Waiting for inference workers to finish...")
        for worker in inference_workers:
            worker.join(timeout=5)

        if git_thread and git_thread.is_alive():
            print("Waiting for the final Git push to complete...")
            git_thread.join()

        print("\n--- Final Save ---", flush=True)
        torch.save(model.state_dict(), "d2cfr_model_final.pth")
        print("Final model saved. Exiting.")

if __name__ == "__main__":
    main()
