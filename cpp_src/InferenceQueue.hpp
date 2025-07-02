#pragma once
#include <vector>
#include <cstdint> // Для uint64_t
#include "concurrentqueue.h"

// Уникальный идентификатор для каждого запроса/ответа
using RequestId = uint64_t;

struct InferenceRequest {
    RequestId id;
    std::vector<float> infoset;
    int num_actions;
};

struct InferenceResponse {
    RequestId id;
    std::vector<float> regrets;
};

// Очередь для запросов от C++ к Python
class InferenceRequestQueue {
public:
    void push(InferenceRequest&& request) {
        queue_.enqueue(std::move(request));
    }

    bool pop_n(std::vector<InferenceRequest>& requests, size_t n) {
        return queue_.try_dequeue_bulk(requests.begin(), n) > 0;
    }

private:
    moodycamel::ConcurrentQueue<InferenceRequest> queue_;
};

// Очередь для ответов от Python к C++
class InferenceResponseQueue {
public:
    void push(InferenceResponse&& response) {
        queue_.enqueue(std::move(response));
    }

    bool pop(InferenceResponse& response) {
        return queue_.try_dequeue(response);
    }

private:
    moodycamel::ConcurrentQueue<InferenceResponse> queue_;
};
