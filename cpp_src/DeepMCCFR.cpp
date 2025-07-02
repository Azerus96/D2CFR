#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace ofc {

DeepMCCFR::DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceRequestQueue* req_q, InferenceResponseQueue* resp_q, std::atomic<bool>* stop_flag, int worker_id) 
    : action_limit_(action_limit), 
      sample_queue_(sample_queue), 
      request_queue_(req_q),
      response_queue_(resp_q),
      stop_flag_(stop_flag), 
      rng_(std::random_device{}()),
      worker_id_(worker_id),
      next_node_id_(0)
{
    local_buffer_.reserve(LOCAL_BUFFER_CAPACITY);
    waiting_nodes_.reserve(4096);
}

DeepMCCFR::~DeepMCCFR() {
    if (!local_buffer_.empty()) {
        flush_local_buffer();
    }
}

void DeepMCCFR::flush_local_buffer() {
    if (local_buffer_.empty()) return;
    SampleBatch batch;
    batch.infosets.reserve(local_buffer_.size());
    batch.regrets.reserve(local_buffer_.size());
    batch.num_actions.reserve(local_buffer_.size());
    for (auto& sample : local_buffer_) {
        batch.infosets.push_back(std::move(sample.infoset_vector));
        batch.regrets.push_back(std::move(sample.regrets_vector));
        batch.num_actions.push_back(sample.num_actions);
    }
    sample_queue_->push(std::move(batch));
    local_buffer_.clear();
}

void DeepMCCFR::run_main_loop() {
    start_new_traversals();

    while (!stop_flag_->load(std::memory_order_relaxed)) {
        process_responses();
        if (waiting_nodes_.empty()) {
            start_new_traversals();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    flush_local_buffer();
}

void DeepMCCFR::start_new_traversals() {
    for (int i = 0; i < 1024; ++i) {
        GameState state;
        int player_to_traverse = (rng_() % 2);
        traverse(std::move(state), player_to_traverse, std::nullopt);
    }
}

void DeepMCCFR::process_responses() {
    InferenceResponse response;
    while(response_queue_->pop(response)) {
        NodeId node_id = response.id;
        auto it = waiting_nodes_.find(node_id);
        if (it == waiting_nodes_.end()) continue;

        WaitingNode waiting_node = std::move(it->second);
        waiting_nodes_.erase(it);

        int num_actions = waiting_node.legal_actions.size();
        std::vector<float> regrets = std::move(response.regrets);
        
        float total_positive_regret = 0.0f;
        for (int i = 0; i < num_actions; ++i) {
            waiting_node.strategy[i] = (regrets[i] > 0) ? regrets[i] : 0.0f;
            total_positive_regret += waiting_node.strategy[i];
        }
        if (total_positive_regret > 0) {
            for (int i = 0; i < num_actions; ++i) waiting_node.strategy[i] /= total_positive_regret;
        } else {
            std::fill(waiting_node.strategy.begin(), waiting_node.strategy.end(), 1.0f / num_actions);
        }

        // Сохраняем узел обратно, теперь он ждет дочерних результатов
        waiting_nodes_[node_id] = std::move(waiting_node);

        // Запускаем дочерние симуляции
        GameState state_template; 
        for (int i = 0; i < num_actions; ++i) {
            GameState next_state = state_template;
            UndoInfo undo_info;
            next_state.apply_action(waiting_nodes_[node_id].legal_actions[i], -1, undo_info);
            traverse(std::move(next_state), -1, ParentInfo{node_id, i});
        }
    }
}

void DeepMCCFR::backtrack_utility(const ParentInfo& parent_info, const std::map<int, float>& utility) {
    auto it = waiting_nodes_.find(parent_info.parent_id);
    if (it == waiting_nodes_.end()) return;

    WaitingNode& parent_node = it->second;
    parent_node.action_utils[parent_info.action_index] = utility;
    parent_node.num_children_remaining--;

    if (parent_node.num_children_remaining == 0) {
        for (size_t i = 0; i < parent_node.legal_actions.size(); ++i) {
            for (auto const& [player_idx, util] : parent_node.action_utils[i]) {
                parent_node.node_util[player_idx] += parent_node.strategy[i] * util;
            }
        }

        std::vector<float> true_regrets(parent_node.legal_actions.size());
        for (size_t i = 0; i < parent_node.legal_actions.size(); ++i) {
            true_regrets[i] = parent_node.action_utils[i][parent_node.current_player] - parent_node.node_util[parent_node.current_player];
        }

        local_buffer_.push_back({std::move(parent_node.infoset_vector), std::move(true_regrets), (int)parent_node.legal_actions.size()});
        if (local_buffer_.size() >= LOCAL_BUFFER_CAPACITY) {
            flush_local_buffer();
        }
        
        waiting_nodes_.erase(it);
    }
}

void DeepMCCFR::traverse(GameState state, int traversing_player, std::optional<ParentInfo> parent_info) {
    if (state.is_terminal()) {
        auto payoffs_pair = state.get_payoffs(evaluator_);
        if (parent_info) {
            // ИСПРАВЛЕНИЕ: Явно преобразуем std::pair в std::map
            std::map<int, float> payoffs_map = {{0, payoffs_pair.first}, {1, payoffs_pair.second}};
            backtrack_utility(*parent_info, payoffs_map);
        }
        return;
    }

    int current_player = state.get_current_player();
    if (traversing_player != -1 && current_player != traversing_player) {
        std::vector<Action> legal_actions;
        state.get_legal_actions(action_limit_, legal_actions, rng_);
        if (legal_actions.empty()) {
             UndoInfo undo_info;
             state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
             traverse(std::move(state), traversing_player, parent_info);
             return;
        }
        int action_idx = std::uniform_int_distribution<int>(0, legal_actions.size() - 1)(rng_);
        UndoInfo undo_info;
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        traverse(std::move(state), traversing_player, parent_info);
        return;
    }

    std::vector<Action> legal_actions;
    state.get_legal_actions(action_limit_, legal_actions, rng_);
    int num_actions = legal_actions.size();

    if (num_actions == 0) {
        UndoInfo undo_info;
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        traverse(std::move(state), traversing_player, parent_info);
        return;
    }

    NodeId node_id = (static_cast<uint64_t>(worker_id_) << 48) | next_node_id_.fetch_add(1);
    std::vector<float> infoset_vec = featurize(state, current_player);

    WaitingNode node;
    node.num_children_remaining = num_actions;
    node.strategy.resize(num_actions);
    node.legal_actions = std::move(legal_actions);
    node.node_util = {{0, 0.0f}, {1, 0.0f}};
    node.action_utils.resize(num_actions);
    node.infoset_vector = infoset_vec;
    node.current_player = current_player;
    
    waiting_nodes_[node_id] = std::move(node);

    request_queue_->push({node_id, std::move(infoset_vec), num_actions});
}

// featurize остается без изменений
std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);
    std::vector<float> features(INFOSET_SIZE, 0.0f);
    
    int offset = 0;
    
    features[offset++] = static_cast<float>(state.get_street());
    features[offset++] = static_cast<float>(state.get_dealer_pos());
    features[offset++] = static_cast<float>(state.get_current_player());
    
    const auto& dealt_cards = state.get_dealt_cards();
    for (Card c : dealt_cards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    
    auto process_board = [&](const Board& board, int& current_offset) {
        for(int i=0; i<3; ++i) {
            Card c = board.top[i];
            if (c != INVALID_CARD) features[current_offset + i*53 + c] = 1.0f;
            else features[current_offset + i*53 + 52] = 1.0f;
        }
        current_offset += 3 * 53;
        
        for(int i=0; i<5; ++i) {
            Card c = board.middle[i];
            if (c != INVALID_CARD) features[current_offset + i*53 + c] = 1.0f;
            else features[current_offset + i*53 + 52] = 1.0f;
        }
        current_offset += 5 * 53;
        
        for(int i=0; i<5; ++i) {
            Card c = board.bottom[i];
            if (c != INVALID_CARD) features[current_offset + i*53 + c] = 1.0f;
            else features[current_offset + i*53 + 52] = 1.0f;
        }
        current_offset += 5 * 53;
    };
    
    process_board(my_board, offset);
    process_board(opp_board, offset);
    
    const auto& my_discards = state.get_my_discards(player_view);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    
    features[offset++] = static_cast<float>(state.get_opponent_discard_count(player_view));
    
    return features;
}

} // namespace ofc
