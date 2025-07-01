#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>

struct InferenceRequest {
    std::vector<float> infoset;
    std::promise<std::vector<float>> promise;
    int num_actions;
};

class InferenceQueue {
public:
    InferenceQueue() : stop_flag_(false) {}

    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(request));
        lock.unlock();
        // --- ГЛАВНОЕ ИЗМЕНЕНИЕ: Будим ВСЕХ, а не одного ---
        // Это позволит всем 8 Python-воркерам конкурировать за запросы.
        cv_.notify_all(); 
    }

    std::vector<InferenceRequest> pop_n(size_t n) {
        std::vector<InferenceRequest> requests;
        std::unique_lock<std::mutex> lock(mtx_);
        
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_.load(); });

        if (stop_flag_.load() && queue_.empty()) {
            return requests;
        }

        size_t count = std::min(n, queue_.size());
        requests.reserve(count);
        std::move(queue_.begin(), queue_.begin() + count, std::back_inserter(requests));
        queue_.erase(queue_.begin(), queue_.begin() + count);
        
        return requests;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_flag_.store(true);
        lock.unlock();
        cv_.notify_all();
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_;
};
