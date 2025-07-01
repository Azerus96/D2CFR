#pragma once
#include <vector>
#include "constants.hpp"
#include "concurrentqueue.h" // Используем новую lock-free библиотеку

namespace ofc {

// Структура, которую C++ воркеры будут класть в очередь
struct SampleBatch {
    // Мы будем перемещать векторы, чтобы избежать копирования
    std::vector<std::vector<float>> infosets;
    std::vector<std::vector<float>> regrets;
    std::vector<int> num_actions;
};

// Обертка над lock-free очередью moodycamel::ConcurrentQueue
class SampleQueue {
public:
    // Операция enqueue здесь lock-free, много потоков могут вызывать ее одновременно без блокировок
    void push(SampleBatch&& batch) {
        queue_.enqueue(std::move(batch));
    }

    // pop будет вызываться из одного Python-потока.
    // Он неблокирующий: возвращает true, если элемент был извлечен, и false, если очередь пуста.
    bool pop(SampleBatch& batch) {
        return queue_.try_dequeue(batch);
    }

private:
    moodycamel::ConcurrentQueue<SampleBatch> queue_;
};

}
