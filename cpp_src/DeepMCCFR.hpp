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
#include <functional>

namespace ofc {

// Тип для утилиты (полезности) для каждого игрока
using Utility = std::map<int, float>;

// Функция "продолжения", которую вызывают, когда результат (утилита) готов
using Continuation = std::function<void(const Utility&)>;

struct LocalSample {
    std::vector<float> infoset_vector;
    std::vector<float> regrets_vector;
    int num_actions;
};

// Структура для состояния, ожидающего ответа от нейросети
struct WaitingForNetwork {
    GameState state;
    int traversing_player;
    Continuation on_complete; // Что делать, когда ответ от сети придет
};

// Структура для состояния, ожидающего результатов от дочерних узлов
struct WaitingForChildren {
    GameState state;
    int traversing_player;
    int current_player;
    std::vector<Action> legal_actions;
    std::vector<float> strategy;
    std::vector<Utility> child_utils;
    std::atomic<int> children_remaining;
    Continuation on_complete; // Что делать, когда все дочерние узлы ответят

    // Явный конструктор для корректной инициализации std::atomic
    WaitingForChildren(
        const GameState& s,
        int tp,
        int cp,
        const std::vector<Action>& la,
        const std::vector<float>& strat,
        std::vector<Utility>&& cu,
        int children_count,
        Continuation&& oc
    ) : state(s),
        traversing_player(tp),
        current_player(cp),
        legal_actions(la),
        strategy(strat),
        child_utils(std::move(cu)),
        children_remaining(children_count),
        on_complete(std::move(oc))
    {}

    // Явно запрещаем копирование и перемещение из-за std::atomic
    WaitingForChildren(const WaitingForChildren&) = delete;
    WaitingForChildren& operator=(const WaitingForChildren&) = delete;
    WaitingForChildren(WaitingForChildren&&) = delete;
    WaitingForChildren& operator=(WaitingForChildren&&) = delete;
};

// Используем std::variant для хранения одного из двух состояний "парковки"
using ParkedTraversal = std::variant<WaitingForNetwork, WaitingForChildren>;

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit, SampleQueue* sample_queue, InferenceRequestQueue* req_q, InferenceResponseQueue* resp_q, std::atomic<bool>* stop_flag, int worker_id);
    ~DeepMCCFR();
    
    void run_main_loop();

private:
    void flush_local_buffer();
    void process_responses();
    void start_new_traversals();

    // Основная функция обхода. Теперь она не возвращает значение, а принимает "продолжение".
    void traverse(GameState& state, int traversing_player, Continuation on_complete);
    
    // Вспомогательная функция для обработки результатов от дочерних узлов
    void on_child_util_ready(RequestId parent_id, int action_index, const Utility& util);

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
    static constexpr int MAX_ACTIVE_TRAVERSALS = 512;

    // Карта для хранения "припаркованных" состояний. Ключ - ID запроса.
    std::unordered_map<RequestId, ParkedTraversal> parked_traversals_;
};

}
