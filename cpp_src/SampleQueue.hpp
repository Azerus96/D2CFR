#pragma once
#include <vector>
#include <thread> // Для std::this_thread::sleep_for
#include <chrono> // Для std::chrono::microseconds
#include "constants.hpp"
#include "concurrentqueue.h" // Используем lock-free библиотеку

namespace ofc {

// Структура, которую C++ воркеры будут класть в очередь
struct SampleBatch {
    std::vector<std::vector<float>> infosets;
    std::vector<std::vector<float>> regrets;
    std::vector<int> num_actions;
};

// Обертка над lock-free очередью moodycamel::ConcurrentQueue
class SampleQueue {
public:
    SampleQueue() {}

    // Операция enqueue здесь lock-free, много потоков могут вызывать ее одновременно без блокировок
    void push(SampleBatch&& batch) {
        queue_.enqueue(std::move(batch));
    }

    // Блокирующий pop, который будет вызываться из Python
    bool pop(SampleBatch& batch) {
        // try_dequeue неблокирующий, поэтому мы должны сами организовать ожидание
        // Это называется "активное ожидание" (spin-wait), оно эффективно, когда ожидание короткое.
        while (!queue_.try_dequeue(batch)) {
            // Если очередь пуста, немного "поспим", чтобы не грузить CPU впустую.
            // Это компромисс между отзывчивостью и нагрузкой на CPU.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return true;
    }

private:
    moodycamel::ConcurrentQueue<SampleBatch> queue_;
};

}
