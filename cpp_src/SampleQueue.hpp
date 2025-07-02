#pragma once
#include <vector>
#include <thread>
#include <chrono>
#include <atomic> // <-- ДОБАВЛЕНО

#include "constants.hpp"
#include "concurrentqueue.h"

namespace ofc {

struct SampleBatch {
    std::vector<std::vector<float>> infosets;
    std::vector<std::vector<float>> regrets;
    std::vector<int> num_actions;
};

// Обертка над lock-free очередью с блокирующим pop
class SampleQueue {
public:
    // ИЗМЕНЕНИЕ: Инициализируем флаг остановки
    SampleQueue() : stop_flag_(false) {}

    void push(SampleBatch&& batch) {
        queue_.enqueue(std::move(batch));
    }

    // ИЗМЕНЕНИЕ: Блокирующий pop с возможностью прерывания
    bool pop(SampleBatch& batch) {
        // try_dequeue неблокирующий, поэтому мы должны сами организовать ожидание
        while (!queue_.try_dequeue(batch)) {
            // Если получили сигнал на остановку и очередь все еще пуста, выходим.
            if (stop_flag_.load(std::memory_order_relaxed)) {
                return false;
            }
            // Если очередь пуста, немного поспим, чтобы не грузить CPU
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return true;
    }

    // ИЗМЕНЕНИЕ: Метод для установки флага остановки
    void stop() {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

private:
    moodycamel::ConcurrentQueue<SampleBatch> queue_;
    std::atomic<bool> stop_flag_; // <-- ДОБАВЛЕНО
};

}
