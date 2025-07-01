#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic> // <-- ДОБАВЛЕНО

struct InferenceRequest {
    std::vector<float> infoset;
    std::promise<std::vector<float>> promise;
    int num_actions;
};

class InferenceQueue {
public:
    // --- ДОБАВЛЕНО: Конструктор для инициализации флага ---
    InferenceQueue() : stop_flag_(false) {}

    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(request));
        lock.unlock();
        cv_.notify_one();
    }

    std::vector<InferenceRequest> pop_n(size_t n) {
        std::vector<InferenceRequest> requests;
        std::unique_lock<std::mutex> lock(mtx_);
        
        // --- ИЗМЕНЕНИЕ: Ждем, пока не появятся данные ИЛИ не будет сигнала остановки ---
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_.load(); });

        // Если нас разбудили сигналом остановки и очередь пуста, возвращаем пустой вектор
        if (stop_flag_.load() && queue_.empty()) {
            return requests;
        }

        size_t count = std::min(n, queue_.size());
        requests.reserve(count);
        std::move(queue_.begin(), queue_.begin() + count, std::back_inserter(requests));
        queue_.erase(queue_.begin(), queue_.begin() + count);
        
        return requests;
    }

    // --- НОВОЕ: Метод для установки флага остановки ---
    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_flag_.store(true);
        lock.unlock();
        // Будим ВСЕ потоки, чтобы они проверили флаг и завершились
        cv_.notify_all();
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_; // <-- ДОБАВЛЕНО: Атомарный флаг для безопасной остановки
};
