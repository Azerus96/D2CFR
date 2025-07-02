#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "InferenceQueue.hpp"
#include "SampleQueue.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <atomic>
#include <unordered_map>
#include <optional>

namespace ofc {

struct LocalSample {
    std::vector<float> infoset_vector;
    std::vector<float> regrets_vector;
    int num_actions;
};

// Возвращаемся к простой структуре для парковки
struct ParkedTraversal {
    GameState state;
    int traversing_player;
};

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceRequestQueue* req_q, InferenceResponseQueue* resp_q, std::atomic<bool>* stop_flag, int worker_id);
    ~DeepMCCFR();
    
    void run_main_loop();

private:
    void flush_local_buffer();
    void process_responses();
    void start_new_traversals();
    // traverse теперь снова возвращает утилиты
    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);

    HandEvaluator evaluator_;
    SampleQueue* sample_queue_;
    InferenceRequestQueue* request_queue_;
    InferenceResponseQueue* response_queue_;
    std::atomic<bool>* stop_flag_;
    size_t action_limit_;
    std::mt19937 rng_;
    int worker_id_;
    std::atomic<uint64_t> next_request_id_;

    std::vector<LocalSample> local_buffer_;
    static constexpr size_t LOCAL_BUFFER_CAPACITY = 256;
    static constexpr int MAX_ACTIVE_TRAVERSALS = 512; // Снизим на всякий случай

    std::unordered_map<RequestId, ParkedTraversal> parked_traversals_;
};

}
