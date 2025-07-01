#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace ofc {

// --- НОВОЕ: Структура для локального буфера ---
// Она будет хранить сэмплы внутри каждого воркера, чтобы избежать частых блокировок
struct LocalSample {
    std::vector<float> infoset_vector;
    std::vector<float> regrets_vector;
    int num_actions;
};

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue);
    // --- ДОБАВЛЕНО: Деструктор для сброса остатков данных при завершении работы ---
    ~DeepMCCFR(); 
    
    void run_traversal();

private:
    // --- ДОБАВЛЕНО: Функция для сброса локального буфера в глобальный ---
    void flush_local_buffer(); 

    HandEvaluator evaluator_;
    SharedReplayBuffer* replay_buffer_; 
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_;

    // --- НОВОЕ: Локальный буфер для каждого воркера ---
    std::vector<LocalSample> local_buffer_;
    // Размер локального буфера. Можно тюнить. 128 - хороший старт.
    static constexpr size_t LOCAL_BUFFER_CAPACITY = 128;
    // ------------------------------------------------

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
