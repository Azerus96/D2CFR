#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>

// Запрос на инференс, который C++ отправляет в Python
struct InferenceRequest {
    std::vector<float> infoset;
    std::promise<std::vector<float>> promise;
    int num_actions;
};

// Потокобезопасная очередь для запросов
class InferenceQueue {
public:
    void push(InferenceRequest&& request) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push_back(std::move(request));
        lock.unlock();
        // Уведомляем ОДИН ожидающий поток, что появились данные.
        // Это эффективнее, чем будить всех (notify_all).
        cv_.notify_one();
    }

    // --- ИЗМЕНЕНИЕ: Забираем не все, а фиксированное количество или меньше ---
    // Этот метод будет вызываться из Python.
    std::vector<InferenceRequest> pop_n(size_t n) {
        std::vector<InferenceRequest> requests;
        std::unique_lock<std::mutex> lock(mtx_);
        
        // Ждем, пока не появится хотя бы один элемент.
        // Поток освобождает мьютекс и "засыпает", пока его не разбудит cv_.notify_one().
        cv_.wait(lock, [this] { return !queue_.empty(); });

        // Определяем, сколько элементов можно забрать (не больше, чем есть в очереди)
        size_t count = std::min(n, queue_.size());
        requests.reserve(count);
        
        // Эффективно перемещаем элементы из начала очереди в наш вектор
        std::move(queue_.begin(), queue_.begin() + count, std::back_inserter(requests));
        
        // Удаляем перемещенные элементы из очереди
        queue_.erase(queue_.begin(), queue_.begin() + count);
        
        return requests;
    }

private:
    std::deque<InferenceRequest> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
};
