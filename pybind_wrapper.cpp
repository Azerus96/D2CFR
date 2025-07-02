#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <atomic>

#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/SampleQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Asynchronous Traversal";

    py::class_<std::atomic<bool>>(m, "AtomicBool")
        .def(py::init<bool>())
        .def("load", [](const std::atomic<bool> &a) { return a.load(); })
        .def("store", [](std::atomic<bool> &a, bool val) { a.store(val); });

    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def_readonly("id", &InferenceRequest::id)
        .def_readonly("infoset", &InferenceRequest::infoset)
        .def_readonly("num_actions", &InferenceRequest::num_actions);

    py::class_<InferenceResponse>(m, "InferenceResponse")
        .def(py::init<RequestId, std::vector<float>>(), py::arg("id"), py::arg("regrets"));

    py::class_<InferenceRequestQueue>(m, "InferenceRequestQueue")
        .def(py::init<>())
        .def("pop_n", [](InferenceRequestQueue& q, size_t n) {
            std::vector<InferenceRequest> reqs;
            // ВАЖНО: pybind11 не может автоматически преобразовать список в вектор для bulk-операций.
            // Нужно создать вектор нужного размера и передать указатели.
            reqs.resize(n);
            size_t dequeued_count = q.pop_n(reqs, n);
            reqs.resize(dequeued_count);
            return reqs;
        }, py::arg("n"), py::call_guard<py::gil_scoped_release>());

    py::class_<InferenceResponseQueue>(m, "InferenceResponseQueue")
        .def(py::init<>())
        // ИСПРАВЛЕНИЕ: Используем лямбду для правильной передачи rvalue-ссылки
        .def("push", [](InferenceResponseQueue &q, InferenceResponse resp) {
            q.push(std::move(resp));
        }, py::call_guard<py::gil_scoped_release>());

    py::class_<ofc::SampleBatch>(m, "SampleBatch")
        .def(py::init<>())
        .def_readonly("infosets", &ofc::SampleBatch::infosets)
        .def_readonly("regrets", &ofc::SampleBatch::regrets)
        .def_readonly("num_actions", &ofc::SampleBatch::num_actions);

    py::class_<ofc::SampleQueue>(m, "SampleQueue")
        .def(py::init<>())
        .def("pop", [](ofc::SampleQueue& q) -> py::object {
            ofc::SampleBatch batch;
            bool success;
            {
                py::gil_scoped_release release;
                success = q.pop(batch);
            }
            if (success) { return py::cast(batch); }
            return py::none();
        })
        .def("stop", &ofc::SampleQueue::stop);

    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions)
        .def("push_batch", [](ofc::SharedReplayBuffer &buffer, const ofc::SampleBatch& batch) {
            for (size_t i = 0; i < batch.infosets.size(); ++i) {
                buffer.push(batch.infosets[i], batch.regrets[i], batch.num_actions[i]);
            }
        }, py::arg("batch"))
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            int action_limit = buffer.get_max_actions();
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto regrets_np = py::array_t<float>(batch_size * action_limit);
            buffer.sample(batch_size, static_cast<float*>(infosets_np.request().ptr), static_cast<float*>(regrets_np.request().ptr));
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            regrets_np.resize({batch_size, action_limit});
            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"));

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SampleQueue*, InferenceRequestQueue*, InferenceResponseQueue*, std::atomic<bool>*, int>(), 
             py::arg("action_limit"), py::arg("sample_queue"), py::arg("request_queue"), py::arg("response_queue"), py::arg("stop_flag"), py::arg("worker_id"))
        .def("run_main_loop", &ofc::DeepMCCFR::run_main_loop, py::call_guard<py::gil_scoped_release>());
}
