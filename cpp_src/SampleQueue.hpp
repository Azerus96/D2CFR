#pragma once
#include <vector>
#include <thread> // Для std::this_thread::sleep_for
#include <chrono> // Для std::chrono::microseconds
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
    SampleQueue() {}

    void push(SampleBatch&& batch) {
        queue_.enqueue(std::move(batch));
    }

    // Блокирующий pop
    bool pop(SampleBatch& batch) {
        // try_dequeue неблокирующий, поэтому мы должны сами организовать ожидание
        while (!queue_.try_dequeue(batch)) {
            // Если очередь пуста, немного поспим, чтобы не грузить CPU
            // Это не идеально, но гораздо проще, чем condition_variable с этой библиотекой
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return true;
    }

private:
    moodycamel::ConcurrentQueue<SampleBatch> queue_;
};

}
