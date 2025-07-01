#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "constants.hpp"

namespace ofc {

// Структура, которую C++ воркеры будут класть в очередь
struct SampleBatch {
    // Мы будем перемещать векторы, чтобы избежать копирования
    std::vector<std::vector<float>> infosets;
    std::vector<std::vector<float>> regrets;
    std::vector<int> num_actions;
};

// Потокобезопасная очередь для готовых сэмплов
class SampleQueue {
public:
    SampleQueue() : stop_flag_(false) {}

    void push(SampleBatch&& batch) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(batch));
        lock.unlock();
        cv_.notify_one();
    }

    SampleBatch pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_.load(); });

        if (stop_flag_.load() && queue_.empty()) {
            return {}; // Возвращаем пустой батч как сигнал завершения
        }

        SampleBatch batch = std::move(queue_.front());
        queue_.pop_front();
        return batch;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        stop_flag_.store(true);
        lock.unlock();
        cv_.notify_all();
    }

private:
    std::deque<SampleBatch> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_;
};

}
