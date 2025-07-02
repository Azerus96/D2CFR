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
cpu_count = os.cpu_count() or 88
# Асинхронная модель менее чувствительна к количеству инференс-воркеров,
# но давайте дадим ей достаточно.
NUM_INFERENCE_WORKERS = 8
NUM_CPP_WORKERS = max(8, cpu_count - NUM_INFERENCE_WORKERS - 8)
NUM_COMPUTATION_THREADS = str(NUM_INFERENCE_WORKERS)

os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

from .model import DuelingNetwork
# ИЗМЕНЕНИЕ: Импортируем новые классы
from ofc_engine import (DeepMCCFR, SharedReplayBuffer, SampleQueue, AtomicBool,
                        InferenceRequestQueue, InferenceResponseQueue, InferenceResponse)

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 100
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 5_000_000
BATCH_SIZE = 2048
SAVE_INTERVAL_SECONDS = 1800
MODEL_PATH = "d2cfr_model.pth"
INFERENCE_BATCH_SIZE = 1024 # Можно увеличить, т.к. инференс теперь не блокирует

class InferenceWorker(threading.Thread):
    def __init__(self, model_provider, req_q, resp_q, device, worker_id):
        super().__init__(daemon=True)
        self.model_provider = model_provider
        self.req_q = req_q
        self.resp_q = resp_q
        self.device = device
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.model = None

    def run(self):
        print(f"InferenceWorker-{self.worker_id} started.", flush=True)
        self.model = self.model_provider.get_model()
        
        while not self.stop_event.is_set():
            try:
                if self.model_provider.is_updated(self.worker_id):
                    self.model = self.model_provider.get_model()
                    self.model_provider.mark_updated(self.worker_id)
                
                requests = self.req_q.pop_n(INFERENCE_BATCH_SIZE)
                if not requests:
                    time.sleep(0.001)
                    continue
                
                self.process_batch(requests)
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error in InferenceWorker-{self.worker_id}: {e}", flush=True)
                    traceback.print_exc()
        print(f"InferenceWorker-{self.worker_id} stopped.", flush=True)

    def process_batch(self, requests):
        ids = [req.id for req in requests]
        infosets = [req.infoset for req in requests]
        num_actions = [req.num_actions for req in requests]

        tensor = torch.tensor(infosets, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            results_tensor = self.model(tensor)
        results_list = results_tensor.cpu().numpy()

        for i in range(len(requests)):
            result = results_list[i][:num_actions[i]].tolist()
            # Отправляем ответ с ID обратно в C++
            self.resp_q.push(InferenceResponse(ids[i], result))

    def stop(self):
        self.stop_event.set()

# ... (InferenceModelProvider и ReplayBufferWriter остаются такими же, как в предыдущей версии) ...
class InferenceModelProvider:
    def __init__(self, model, device, num_workers):
        self.device = device
        self.num_workers = num_workers
        self.model_lock = threading.Lock()
        self.updated_flags = [True] * num_workers
        self.stop_event = threading.Event()
        self.set_model(model)

    def set_model(self, model):
        with self.model_lock:
            self.model = torch.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
            self.updated_flags = [True] * self.num_workers

    def get_model(self):
        with self.model_lock:
            return self.model

    def is_updated(self, worker_id):
        return self.updated_flags[worker_id]

    def mark_updated(self, worker_id):
        self.updated_flags[worker_id] = False
    
    def stop(self):
        self.stop_event.set()

class ReplayBufferWriter(threading.Thread):
    def __init__(self, sample_queue, replay_buffer):
        super().__init__(daemon=True)
        self.sample_queue = sample_queue
        self.replay_buffer = replay_buffer
        self.stop_event = threading.Event()

    def run(self):
        print(f"ReplayBufferWriter started.", flush=True)
        while not self.stop_event.is_set():
            batch = self.sample_queue.pop()
            if batch is not None:
                self.replay_buffer.push_batch(batch)
            elif self.stop_event.is_set():
                break
        print(f"ReplayBufferWriter stopped.", flush=True)

    def stop(self):
        self.stop_event.set()
        self.sample_queue.stop()

# ... (push_to_github без изменений) ...
def push_to_github(model_path, commit_message):
    try:
        print("Pushing progress to GitHub...", flush=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True, capture_output=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True, capture_output=True)
        subprocess.run(['git', 'add', model_path], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        print("Progress pushed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
        print(f"Git stderr: {e.stderr.decode()}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Could not load state_dict. Error: {e}. Starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    sample_queue = SampleQueue()
    request_queue = InferenceRequestQueue()
    response_queue = InferenceResponseQueue()

    model_provider = InferenceModelProvider(model, device, NUM_INFERENCE_WORKERS)
    
    print("Starting Python workers...", flush=True)
    inference_workers = [InferenceWorker(model_provider, request_queue, response_queue, device, i) for i in range(NUM_INFERENCE_WORKERS)]
    replay_buffer_writer = ReplayBufferWriter(sample_queue, replay_buffer)
    for worker in inference_workers:
        worker.start()
    replay_buffer_writer.start()

    stop_flag = AtomicBool(False)
    solvers = [DeepMCCFR(ACTION_LIMIT, sample_queue, request_queue, response_queue, stop_flag, i) for i in range(NUM_CPP_WORKERS)]
    
    git_thread = None
    futures = set()
    executor = ThreadPoolExecutor(max_workers=NUM_CPP_WORKERS)

    try:
        print(f"Submitting {NUM_CPP_WORKERS} C++ worker tasks...", flush=True)
        for s in solvers:
            futures.add(executor.submit(s.run_main_loop))
        
        print(f"Warm-up phase: waiting for Replay Buffer to contain at least {BATCH_SIZE} samples.", flush=True)
        while replay_buffer.get_count() < BATCH_SIZE:
            for future in list(futures):
                if future.done() and future.exception():
                    print("\nFATAL: C++ worker failed during warm-up!", flush=True)
                    raise future.exception()
            
            print(f"\rBuffer filling: {replay_buffer.get_count()}/{BATCH_SIZE}", end="", flush=True)
            time.sleep(1)
        
        print("\nWarm-up complete. Starting main training loop.", flush=True)

        last_save_time = time.time()
        last_report_time = time.time()
        total_samples_at_last_report = replay_buffer.get_count()
        training_steps = 0

        while True:
            for future in list(futures):
                if future.done() and future.exception():
                    raise future.exception()

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
            training_steps += 1
            
            if training_steps % 100 == 0: # Обновляем модель для инференса реже
                model_provider.set_model(model)
            
            now = time.time()
            if now - last_report_time > 10.0:
                current_buffer_size = replay_buffer.get_count()
                duration = now - last_report_time
                samples_generated_interval = current_buffer_size - total_samples_at_last_report
                samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                
                print(f"\n--- Stats (Step {training_steps}) ---", flush=True)
                print(f"  Loss: {loss.item():.6f}", flush=True)
                print(f"  Sample generation rate: {samples_per_sec:.2f} samples/s", flush=True)
                print(f"  Replay Buffer: {current_buffer_size:,}/{REPLAY_BUFFER_CAPACITY:,} ({current_buffer_size/REPLAY_BUFFER_CAPACITY:.1%})", flush=True)
                
                total_samples_at_last_report = current_buffer_size
                last_report_time = now

            if now - last_save_time > SAVE_INTERVAL_SECONDS:
                if git_thread and git_thread.is_alive():
                    print("Previous Git push is still running. Skipping this save.", flush=True)
                else:
                    print("\n--- Saving model and pushing to GitHub ---", flush=True)
                    torch.save(model.state_dict(), MODEL_PATH)
                    commit_message = f"Training step {training_steps}. Loss: {loss.item():.6f}"
                    
                    git_thread = threading.Thread(target=push_to_github, args=(MODEL_PATH, commit_message))
                    git_thread.start()
                    
                    last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("Stopping all workers...", flush=True)
        stop_flag.store(True)
        
        model_provider.stop()
        for worker in inference_workers:
            worker.stop()
        
        replay_buffer_writer.stop()
        
        if futures:
            print("Shutting down C++ worker pool...")
            executor.shutdown(wait=True, cancel_futures=False)

        print("Waiting for Python workers to finish...")
        replay_buffer_writer.join(timeout=5)
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
