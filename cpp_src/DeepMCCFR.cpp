#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <thread>

namespace ofc {

DeepMCCFR::DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceRequestQueue* req_q, InferenceResponseQueue* resp_q, std::atomic<bool>* stop_flag, int worker_id) 
    : action_limit_(action_limit), 
      sample_queue_(sample_queue), 
      request_queue_(req_q),
      response_queue_(resp_q),
      stop_flag_(stop_flag), 
      rng_(std::random_device{}()),
      worker_id_(worker_id),
      next_request_id_(0)
{
    local_buffer_.reserve(LOCAL_BUFFER_CAPACITY);
    parked_traversals_.reserve(MAX_ACTIVE_TRAVERSALS);
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
    while (!stop_flag_->load(std::memory_order_relaxed)) {
        process_responses();
        start_new_traversals();
        // Если нет активных обходов, немного подождать, чтобы не загружать CPU
        if (parked_traversals_.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    flush_local_buffer();
}

void DeepMCCFR::process_responses() {
    InferenceResponse response;
    while(response_queue_->pop(response)) {
        auto it = parked_traversals_.find(response.id);
        if (it == parked_traversals_.end()) continue;

        // Убедимся, что мы обрабатываем состояние, которое ждало ответа от сети
        WaitingForNetwork* waiting_state_ptr = std::get_if<WaitingForNetwork>(&it->second);
        if (!waiting_state_ptr) continue;

        WaitingForNetwork waiting_state = std::move(*waiting_state_ptr);
        
        // Теперь узел переходит в состояние ожидания дочерних результатов
        GameState state = std::move(waiting_state.state);
        int traversing_player = waiting_state.traversing_player;
        Continuation on_complete = std::move(waiting_state.on_complete);

        std::vector<Action> legal_actions;
        state.get_legal_actions(action_limit_, legal_actions, rng_);
        int num_actions = legal_actions.size();

        std::vector<float> regrets = std::move(response.regrets);
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

        // Создаем новое состояние "ожидания детей"
        parked_traversals_[response.id] = WaitingForChildren{
            state, // Копируем состояние для информации
            traversing_player,
            state.get_current_player(),
            legal_actions,
            strategy,
            std::vector<Utility>(num_actions),
            std::atomic<int>(num_actions),
            std::move(on_complete)
        };

        // Запускаем обходы для каждого действия
        for (int i = 0; i < num_actions; ++i) {
            GameState next_state = state;
            UndoInfo undo_info;
            next_state.apply_action(legal_actions[i], traversing_player, undo_info);
            
            // Создаем "продолжение" для этого дочернего узла.
            // Оно вызовет on_child_util_ready, когда результат будет готов.
            Continuation child_cont = [this, parent_id = response.id, action_idx = i](const Utility& util) {
                this->on_child_util_ready(parent_id, action_idx, util);
            };
            
            traverse(next_state, traversing_player, std::move(child_cont));
        }
    }
}

void DeepMCCFR::on_child_util_ready(RequestId parent_id, int action_index, const Utility& util) {
    auto it = parked_traversals_.find(parent_id);
    if (it == parked_traversals_.end()) return;

    WaitingForChildren* parent_state_ptr = std::get_if<WaitingForChildren>(&it->second);
    if (!parent_state_ptr) return;

    parent_state_ptr->child_utils[action_index] = util;
    
    // Если это был последний дочерний узел, который мы ждали
    if (parent_state_ptr->children_remaining.fetch_sub(1) == 1) {
        WaitingForChildren parent_state = std::move(*parent_state_ptr);
        parked_traversals_.erase(it);

        Utility node_util = {{0, 0.0f}, {1, 0.0f}};
        for(size_t i = 0; i < parent_state.legal_actions.size(); ++i) {
            for(auto const& [player_idx, u] : parent_state.child_utils[i]) {
                node_util[player_idx] += parent_state.strategy[i] * u;
            }
        }

        std::vector<float> true_regrets(parent_state.legal_actions.size());
        for(size_t i = 0; i < parent_state.legal_actions.size(); ++i) {
            true_regrets[i] = parent_state.child_utils[i][parent_state.current_player] - node_util[parent_state.current_player];
        }
        
        std::vector<float> infoset_vec = featurize(parent_state.state, parent_state.current_player);
        local_buffer_.push_back({std::move(infoset_vec), std::move(true_regrets), (int)parent_state.legal_actions.size()});
        if (local_buffer_.size() >= LOCAL_BUFFER_CAPACITY) {
            flush_local_buffer();
        }

        // Вызываем "продолжение" родительского узла, передавая ему вычисленную утилиту
        if (parent_state.on_complete) {
            parent_state.on_complete(node_util);
        }
    }
}


void DeepMCCFR::start_new_traversals() {
    while (parked_traversals_.size() < MAX_ACTIVE_TRAVERSALS) {
        GameState state;
        int player_to_traverse = (rng_() % 2);
        // Для корневого вызова "продолжение" пустое, т.к. результат никуда не передается
        traverse(state, player_to_traverse, nullptr);
    }
}

void DeepMCCFR::traverse(GameState& state, int traversing_player, Continuation on_complete) {
    if (state.is_terminal()) {
        auto payoffs_pair = state.get_payoffs(evaluator_);
        if (on_complete) {
            on_complete({{0, payoffs_pair.first}, {1, payoffs_pair.second}});
        }
        return;
    }

    int current_player = state.get_current_player();
    std::vector<Action> legal_actions;
    state.get_legal_actions(action_limit_, legal_actions, rng_);
    int num_actions = legal_actions.size();
    UndoInfo undo_info;

    if (num_actions == 0) {
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        // Просто продолжаем обход, передавая то же самое "продолжение"
        traverse(state, traversing_player, std::move(on_complete));
        return;
    }

    // Если ход оппонента, семплируем одно действие и продолжаем обход
    if (current_player != traversing_player) {
        int action_idx = std::uniform_int_distribution<int>(0, num_actions - 1)(rng_);
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        traverse(state, traversing_player, std::move(on_complete));
        return;
    }

    // Если наш ход, отправляем запрос в сеть
    RequestId req_id = (static_cast<uint64_t>(worker_id_) << 48) | next_request_id_.fetch_add(1);
    std::vector<float> infoset_vec = featurize(state, current_player);
    
    // "Паркуем" состояние, сохраняя его и "продолжение", которое нужно будет вызвать
    parked_traversals_[req_id] = WaitingForNetwork{state, traversing_player, std::move(on_complete)};
    
    request_queue_->push({req_id, std::move(infoset_vec), num_actions});
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
