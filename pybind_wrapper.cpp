#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/SampleQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Lock-Free Queues";

    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("pop_n", &InferenceQueue::pop_n, py::arg("n"), py::call_guard<py::gil_scoped_release>())
        .def("stop", &InferenceQueue::stop);

    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def_readonly("infoset", &InferenceRequest::infoset)
        .def_readonly("num_actions", &InferenceRequest::num_actions)
        .def("set_result", [](InferenceRequest &req, std::vector<float> result) {
            req.promise.set_value(result);
        });

    py::class_<ofc::SampleBatch>(m, "SampleBatch")
        .def(py::init<>())
        .def_readonly("infosets", &ofc::SampleBatch::infosets)
        .def_readonly("regrets", &ofc::SampleBatch::regrets)
        .def_readonly("num_actions", &ofc::SampleBatch::num_actions);

    py::class_<ofc::SampleQueue>(m, "SampleQueue")
        .def(py::init<>())
        // --- ИЗМЕНЕНИЕ: Явно указываем тип возвращаемого значения лямбды ---
        .def("pop", [](ofc::SampleQueue& q) -> py::object { // <-- Указываем, что возвращаем py::object
            ofc::SampleBatch batch;
            if (q.pop(batch)) {
                return py::cast(batch);
            }
            return py::none(); // py::none() автоматически приводится к py::object
        }, py::call_guard<py::gil_scoped_release>());
    // --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(regrets_np.request().ptr)
            );
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            regrets_np.resize({batch_size, action_limit});
            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"));

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SampleQueue*, InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("sample_queue"), py::arg("inference_queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
