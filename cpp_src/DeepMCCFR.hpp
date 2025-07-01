#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "InferenceQueue.hpp"
#include "SampleQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace ofc {

struct LocalSample {
    std::vector<float> infoset_vector;
    std::vector<float> regrets_vector;
    int num_actions;
};

class DeepMCCFR {
public:
    // Конструктор принимает указатель на новую очередь для сэмплов
    DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceQueue* inference_queue);
    ~DeepMCCFR(); 
    
    void run_traversal();

private:
    void flush_local_buffer(); 

    HandEvaluator evaluator_;
    SampleQueue* sample_queue_;
    InferenceQueue* inference_queue_;
    size_t action_limit_;
    std::mt19937 rng_;

    std::vector<LocalSample> local_buffer_;
    static constexpr size_t LOCAL_BUFFER_CAPACITY = 256;

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
