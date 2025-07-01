#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace ofc {

DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue) 
    : action_limit_(action_limit), replay_buffer_(buffer), inference_queue_(queue), rng_(std::random_device{}()) {
    // Инициализируем локальный буфер с预留空间 для эффективности
    local_buffer_.reserve(LOCAL_BUFFER_CAPACITY);
}

// --- НОВОЕ: Деструктор, который гарантирует, что все данные будут сохранены при остановке ---
DeepMCCFR::~DeepMCCFR() {
    if (!local_buffer_.empty()) {
        flush_local_buffer();
    }
}

// --- НОВОЕ: Функция сброса локального буфера в глобальный ---
void DeepMCCFR::flush_local_buffer() {
    // Этот метод вызывается редко, поэтому блокировка здесь не страшна
    for (const auto& sample : local_buffer_) {
        replay_buffer_->push(sample.infoset_vector, sample.regrets_vector, sample.num_actions);
    }
    local_buffer_.clear();
}

void DeepMCCFR::run_traversal() {
    GameState state; 
    traverse(state, 0);
    state.reset(); 
    traverse(state, 1);
}

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
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 3 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.middle[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.bottom[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
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

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    std::vector<Action> legal_actions;
    state.get_legal_actions(action_limit_, legal_actions, rng_);
    
    int num_actions = legal_actions.size();
    UndoInfo undo_info;

    if (num_actions == 0) {
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = std::uniform_int_distribution<int>(0, num_actions - 1)(rng_);
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    std::vector<float> infoset_vec = featurize(state, traversing_player);
    
    std::vector<float> regrets;
    {
        std::promise<std::vector<float>> promise;
        std::future<std::vector<float>> future = promise.get_future();

        InferenceRequest request;
        request.infoset = infoset_vec;
        request.promise = std::move(promise);
        request.num_actions = num_actions;
        inference_queue_->push(std::move(request));

        regrets = future.get();
    }

    std::vector<float> strategy(num_actions);
    float total_positive_regret = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        strategy[i] = (regrets[i] > 0) ? regrets[i] : 0.0f;
        total_positive_regret += strategy[i];
    }

    if (total_positive_regret > 0) {
        for (int i = 0; i < num_actions; ++i) strategy[i] /= total_positive_regret;
    } else {
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    std::vector<std::map<int, float>> action_utils(num_actions);
    std::map<int, float> node_util = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        state.apply_action(legal_actions[i], traversing_player, undo_info);
        action_utils[i] = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils[i]) {
            node_util[player_idx] += strategy[i] * util;
        }
    }

    std::vector<float> true_regrets(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets[i] = action_utils[i][current_player] - node_util[current_player];
    }
    
    // --- ИЗМЕНЕНИЕ: Сохраняем в локальный буфер ---
    local_buffer_.push_back({infoset_vec, true_regrets, num_actions});
    if (local_buffer_.size() >= LOCAL_BUFFER_CAPACITY) {
        flush_local_buffer();
    }
    // ---------------------------------------------

    return node_util;
}

}
