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
#include <variant>
#include <optional>

namespace ofc {

struct LocalSample {
    std::vector<float> infoset_vector;
    std::vector<float> regrets_vector;
    int num_actions;
};

// Уникальный ID для каждого узла в дереве игры
using NodeId = uint64_t;

// Структура для узла, ожидающего результатов от дочерних узлов
struct WaitingNode {
    int num_children_remaining;
    std::vector<float> strategy;
    std::vector<Action> legal_actions;
    std::map<int, float> node_util;
    std::vector<std::map<int, float>> action_utils;
    std::vector<float> infoset_vector;
    int current_player;
};

// Структура для хранения информации о родительском узле
struct ParentInfo {
    NodeId parent_id;
    int action_index; // Индекс действия, которое привело к дочернему узлу
};

// Результат обхода может быть либо финальным payoff, либо информацией о родителе
using TraversalResult = std::variant<std::map<int, float>, ParentInfo>;

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceRequestQueue* req_q, InferenceResponseQueue* resp_q, std::atomic<bool>* stop_flag, int worker_id);
    ~DeepMCCFR();
    
    void run_main_loop();

private:
    void flush_local_buffer();
    void process_responses();
    void start_new_traversals();
    void traverse(GameState state, int traversing_player, std::optional<ParentInfo> parent_info);
    void backtrack_utility(const ParentInfo& parent_info, const std::map<int, float>& utility);
    std::vector<float> featurize(const GameState& state, int player_view);

    HandEvaluator evaluator_;
    SampleQueue* sample_queue_;
    InferenceRequestQueue* request_queue_;
    InferenceResponseQueue* response_queue_;
    std::atomic<bool>* stop_flag_;
    size_t action_limit_;
    std::mt19937 rng_;
    int worker_id_;
    std::atomic<uint64_t> next_node_id_;

    std::vector<LocalSample> local_buffer_;
    static constexpr size_t LOCAL_BUFFER_CAPACITY = 256;
    
    // Карта для узлов, ожидающих инференса или дочерних результатов
    std::unordered_map<NodeId, WaitingNode> waiting_nodes_;
};

}
