//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "mediapipegraphdefinition.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <pybind11/embed.h> // everything needed for embedding
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../ov_utils.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphexecutor.hpp"
#include "src/mediapipe_calculators/python_backend_calculator.pb.h"

namespace py = pybind11;

namespace ovms {
MediapipeGraphConfig MediapipeGraphDefinition::MGC;

const std::string MediapipeGraphDefinition::SCHEDULER_CLASS_NAME{"Mediapipe"};

MediapipeGraphDefinition::~MediapipeGraphDefinition(){
    // pybind requires to acquire gil when destructing objects
    if(!this->pythonNodeStates.empty()){
        py::gil_scoped_acquire acquire;
        this->pythonNodeStates.clear();
    }
};

const tensor_map_t MediapipeGraphDefinition::getInputsInfo() const {
    std::shared_lock lock(metadataMtx);
    return this->inputsInfo;
}

const tensor_map_t MediapipeGraphDefinition::getOutputsInfo() const {
    std::shared_lock lock(metadataMtx);
    return this->outputsInfo;
}

Status MediapipeGraphDefinition::validateForConfigFileExistence() {
    std::ifstream ifs(this->mgconfig.getGraphPath());
    if (!ifs.is_open()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to open mediapipe graph definition: {}, file: {}\n", this->getName(), this->mgconfig.getGraphPath());
        return StatusCode::FILE_INVALID;
    }
    this->chosenConfig.clear();
    ifs.seekg(0, std::ios::end);
    this->chosenConfig.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    this->chosenConfig.assign((std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validateForConfigLoadableness() {
    if (chosenConfig.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse empty mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }

    bool success = ::google::protobuf::TextFormat::ParseFromString(chosenConfig, &this->config);
    if (!success) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::dryInitializeTest() {
    ::mediapipe::CalculatorGraph graph;
    auto absStatus = graph.Initialize(this->config);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph: {} initialization failed with message: {}. Check if all required calculators are registered in OVMS", this->getName(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}
Status MediapipeGraphDefinition::validate(ModelManager& manager) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of mediapipe: {}", getName());
    ValidationResultNotifier notifier(this->status, this->loadedNotify);
    if (manager.modelExists(this->getName()) || manager.pipelineDefinitionExists(this->getName())) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph name: {} is already occupied by model or pipeline.", this->getName());
        return StatusCode::MEDIAPIPE_GRAPH_NAME_OCCUPIED;
    }
    Status validationResult = validateForConfigFileExistence();
    if (!validationResult.ok()) {
        return validationResult;
    }
    validationResult = validateForConfigLoadableness();
    if (!validationResult.ok()) {
        return validationResult;
    }
    // TODO
    // 3 validate 1<= outputs
    // 4 validate 1<= inputs
    // 5 validate no side_packets? push into executor check params vs expected side packets
    ::mediapipe::CalculatorGraphConfig proto;
    std::unique_lock lock(metadataMtx);
    auto status = createInputsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create inputs info for mediapipe graph definition: {}", getName());
        return status;
    }
    status = createOutputsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create outputs info for mediapipe graph definition: {}", getName());
        return status;
    }
    status = createInputSidePacketsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create input side packets info for mediapipe graph definition: {}", getName());
        return status;
    }
    // Detect what deserialization needs to be performed
    status = this->setStreamTypes();
    if (!status.ok()) {
        return status;
    }
    // here we will not be available if calculator does not exist in OVMS
    status = this->dryInitializeTest();
    if (!status.ok()) {
        return status;
    }
    lock.unlock();
    notifier.passed = true;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of mediapipe: {}", getName());
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} inputs: {}", getName(), getTensorMapString(inputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} outputs: {}", getName(), getTensorMapString(outputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} kfs pass through: {}", getName(), this->passKfsRequestFlag);
    return StatusCode::OK;
}

MediapipeGraphDefinition::MediapipeGraphDefinition(const std::string name,
    const MediapipeGraphConfig& config,
    MetricRegistry* registry,
    const MetricConfig* metricConfig) :
    name(name),
    status(SCHEDULER_CLASS_NAME, this->name) {
    mgconfig = config;
    passKfsRequestFlag = false;
}

Status MediapipeGraphDefinition::createInputsInfo() {
    inputsInfo.clear();
    inputNames.clear();
    inputNames.reserve(this->config.input_stream().size());
    for (auto& name : config.input_stream()) {
        std::string streamName = MediapipeGraphDefinition::getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph inputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME;
        }
        const auto [it, success] = inputsInfo.insert({streamName, TensorInfo::getUnspecifiedTensorInfo()});
        if (!success) {
            SPDLOG_ERROR("Creating Mediapipe graph inputs name failed for: {}. Input with the same name already exists.", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME;
        }
        inputNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::createInputSidePacketsInfo() {
    inputSidePacketNames.clear();
    for (auto& name : config.input_side_packet()) {
        std::string streamName = MediapipeGraphDefinition::getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph input side packet name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_SIDE_PACKET_STREAM_PACKET_NAME;
        }
        inputSidePacketNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::createOutputsInfo() {
    outputsInfo.clear();
    outputNames.clear();
    outputNames.reserve(this->config.output_stream().size());
    for (auto& name : this->config.output_stream()) {
        std::string streamName = MediapipeGraphDefinition::getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph outputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME;
        }
        const auto [it, success] = outputsInfo.insert({streamName, TensorInfo::getUnspecifiedTensorInfo()});
        if (!success) {
            SPDLOG_ERROR("Creating Mediapipe graph outputs name failed for: {}. Output with the same name already exists.", name);
            return StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME;
        }
        outputNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::create(std::shared_ptr<MediapipeGraphExecutor>& pipeline, const KFSRequest* request, KFSResponse* response) {
    std::unique_ptr<MediapipeGraphDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to execute mediapipe graph: {} since it is not available", getName());
        return status;
    }
    SPDLOG_DEBUG("Creating Mediapipe graph executor: {}", getName());

    pipeline = std::make_shared<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()),
        this->config, this->inputTypes, this->outputTypes, this->inputNames, this->outputNames,  this->pythonNodeStates);
    return status;
}

const std::string KFS_REQUEST_PREFIX{"REQUEST"};
const std::string KFS_RESPONSE_PREFIX{"RESPONSE"};
const std::string MP_TENSOR_PREFIX{"TENSOR"};
const std::string TF_TENSOR_PREFIX{"TFTENSOR"};
const std::string TFLITE_TENSOR_PREFIX{"TFLITE_TENSOR"};
const std::string OV_TENSOR_PREFIX{"OVTENSOR"};
const std::string MP_IMAGE_PREFIX{"IMAGE"};

Status MediapipeGraphDefinition::setStreamTypes() {
    this->inputTypes.clear();
    this->outputTypes.clear();
    this->passKfsRequestFlag = false;
    if (!this->config.input_stream().size() ||
        !this->config.output_stream().size()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph: {}; having less than one input or output is disallowed", getName());
        // validation is incomplete in case this error is triggered
        return StatusCode::INTERNAL_ERROR;
    }
    for (auto& inputStreamName : this->config.input_stream()) {
        inputTypes.emplace(getStreamNamePair(inputStreamName));
    }
    for (auto& outputStreamName : this->config.output_stream()) {
        outputTypes.emplace(getStreamNamePair(outputStreamName));
    }
    bool anyInputTfLite = std::any_of(inputTypes.begin(), inputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::TFLITETENSOR;
    });
    bool anyOutputTfLite = std::any_of(outputTypes.begin(), outputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::TFLITETENSOR;
    });
    if (anyInputTfLite || anyOutputTfLite) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "There is no support for TfLiteTensor deserialization & serialization");
        return StatusCode::NOT_IMPLEMENTED;
    }
    bool kfsRequestPass = std::any_of(inputTypes.begin(), inputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::KFS_REQUEST;
    });
    bool kfsResponsePass = std::any_of(outputTypes.begin(), outputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::KFS_RESPONSE;
    });
    if (kfsRequestPass) {
        if (!kfsResponsePass) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph configuration: {}; KFS passthrough mode is misconfigured. KServe for mediapipe graph passing whole KFS request and response requires: {} tag in the output stream name", getName(), KFS_RESPONSE_PREFIX);
            return Status(StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_OUTPUT_RESPONSE_TAG);

        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "KServe for mediapipe graph: {}; passing whole KFS request graph detected.", getName());
        }
    } else if (kfsResponsePass) {
        if (!kfsRequestPass) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph configuration: {}; KServe for mediapipe graph passing whole KFS request and response requires: {} tag in the input stream name", getName(), KFS_REQUEST_PREFIX);
            return Status(StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_INPUT_REQUEST_TAG);
        }
    }
    if (kfsRequestPass == true) {
        if (this->config.output_stream().size() != 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "KServe passthrough through mediapipe graph requires having only one output(response)");
            return StatusCode::MEDIAPIPE_KFS_PASS_WRONG_OUTPUT_STREAM_COUNT;
        }
        if (this->config.input_stream().size() != 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "KServe passthrough through mediapipe graph requires having only one input (request)");
            return StatusCode::MEDIAPIPE_KFS_PASS_WRONG_INPUT_STREAM_COUNT;
        }
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::reload(ModelManager& manager, const MediapipeGraphConfig& config) {
    // block creating new unloadGuards
    this->status.handle(ReloadEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->mgconfig = config;
    return validate(manager);
}

void MediapipeGraphDefinition::retire(ModelManager& manager) {
    this->status.handle(RetireEvent());
}

bool MediapipeGraphDefinition::isReloadRequired(const MediapipeGraphConfig& config) const {
    if (getStateCode() == PipelineDefinitionStateCode::RETIRED) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reloading previously retired mediapipe definition: {}", getName());
        return true;
    }
    return getMediapipeGraphConfig().isReloadRequired(config);
}

Status MediapipeGraphDefinition::waitForLoaded(std::unique_ptr<MediapipeGraphDefinitionUnloadGuard>& unloadGuard, const uint waitForLoadedTimeoutMicroseconds) {
    unloadGuard = std::make_unique<MediapipeGraphDefinitionUnloadGuard>(*this);

    const uint waitLoadedTimestepMicroseconds = 1000;
    const uint waitCheckpoints = waitForLoadedTimeoutMicroseconds / waitLoadedTimestepMicroseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    std::mutex cvMtx;
    std::unique_lock<std::mutex> cvLock(cvMtx);
    while (waitCheckpointsCounter-- != 0) {
        if (status.isAvailable()) {
            SPDLOG_DEBUG("Successfully waited for mediapipe definition: {}", getName());
            return StatusCode::OK;
        }
        unloadGuard.reset();
        if (!status.canEndLoaded()) {
            if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended due to timeout.", getName());
                return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET;
            } else {
                SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended since it failed to load.", getName());
                return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE;
            }
        }
        SPDLOG_DEBUG("Waiting for available state for mediapipe: {}, with timestep: {}us timeout: {}us check count: {}",
            getName(), waitLoadedTimestepMicroseconds, waitForLoadedTimeoutMicroseconds, waitCheckpointsCounter);
        loadedNotify.wait_for(cvLock,
            std::chrono::microseconds(waitLoadedTimestepMicroseconds),
            [this]() {
                return this->status.isAvailable() ||
                       !this->status.canEndLoaded();
            });
        unloadGuard = std::make_unique<MediapipeGraphDefinitionUnloadGuard>(*this);
    }
    if (!status.isAvailable()) {
        if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
            SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended due to timeout.", getName());
            return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET;
        } else {
            SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended since it failed to load.", getName());
            return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_DEBUG("Succesfully waited for mediapipe definition: {}", getName());
    return StatusCode::OK;
}

const std::string EMPTY_STREAM_NAME{""};

std::string MediapipeGraphDefinition::getStreamName(const std::string& streamFullName) {
    std::vector<std::string> tokens = tokenize(streamFullName, ':');
    if (tokens.size() == 2) {
        return tokens[1];
    } else if (tokens.size() == 1) {
        return tokens[0];
    }
    return EMPTY_STREAM_NAME;
}

std::pair<std::string, mediapipe_packet_type_enum> MediapipeGraphDefinition::getStreamNamePair(const std::string& streamFullName) {
    static std::unordered_map<std::string, mediapipe_packet_type_enum> prefix2enum{
        {KFS_REQUEST_PREFIX, mediapipe_packet_type_enum::KFS_REQUEST},
        {KFS_RESPONSE_PREFIX, mediapipe_packet_type_enum::KFS_RESPONSE},
        {TF_TENSOR_PREFIX, mediapipe_packet_type_enum::TFTENSOR},
        {TFLITE_TENSOR_PREFIX, mediapipe_packet_type_enum::TFLITETENSOR},
        {OV_TENSOR_PREFIX, mediapipe_packet_type_enum::OVTENSOR},
        {MP_TENSOR_PREFIX, mediapipe_packet_type_enum::MPTENSOR},
        {MP_IMAGE_PREFIX, mediapipe_packet_type_enum::MEDIAPIPE_IMAGE}};
    std::vector<std::string> tokens = tokenize(streamFullName, ':');
    // MP convention
    // input_stream: "lowercase_input_stream_name"
    // input_stream: "PACKET_TAG:lowercase_input_stream_name"
    // input_stream: "PACKET_TAG:[0-9]:lowercase_input_stream_name"
    if (tokens.size() == 2 || tokens.size() == 3) {
        auto it = std::find_if(prefix2enum.begin(), prefix2enum.end(), [tokens](const auto& p) {
            const auto& [k, v] = p;
            bool b = startsWith(tokens[0], k);
            return b;
        });
        size_t inputStreamIndex = tokens.size() - 1;
        if (it != prefix2enum.end()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting input stream: {} packet type: {} from: {}", tokens[inputStreamIndex], it->first, streamFullName);
            return {tokens[inputStreamIndex], it->second};
        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting input stream: {} packet type: {} from: {}", tokens[inputStreamIndex], "UNKNOWN", streamFullName);
            return {tokens[inputStreamIndex], mediapipe_packet_type_enum::UNKNOWN};
        }
    } else if (tokens.size() == 1) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting input stream: {} packet type: {} from: {}", tokens[0], "UNKNOWN", streamFullName);
        return {tokens[0], mediapipe_packet_type_enum::UNKNOWN};
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "setting input stream: {} packet type: {} from: {}", "", "UNKNOWN", streamFullName);
    return {"", mediapipe_packet_type_enum::UNKNOWN};
}

Status MediapipeGraphDefinition::initializeNodes() {
    SPDLOG_INFO("MediapipeGraphDefinition initializing graph nodes");
    for (int i = 0; i < config.node().size(); i++){
        if (config.node(i).node_options().size()) {
            mediapipe::PythonBackendCalculatorOptions options;
            config.node(i).node_options(0).UnpackTo(&options);
            const std::string handler_path = options.handler_path();

            if (!std::filesystem::exists(handler_path)){
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Graph: {} error. Python node file: {} does not exist. ", this->name, handler_path);
                return StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST;
            }

            auto fs_handler_path = std::filesystem::path(handler_path);
            fs_handler_path.replace_extension();

            std::string parent_path = fs_handler_path.parent_path();
            std::string filename = fs_handler_path.filename();

            if (this->pythonNodeStates.find(filename) != this->pythonNodeStates.end()) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Python node file: {} already used in graph: {}. ", filename, this->name);
                return StatusCode::PYTHON_NODE_NAME_ALREADY_EXISTS;
            }

            try {
                py::gil_scoped_acquire acquire;
                py::module_ sys = py::module_::import("sys");
                sys.attr("path").attr("append")(parent_path.c_str());
                py::module_ script = py::module_::import(filename.c_str());
                py::object OvmsPythonModel = script.attr("OvmsPythonModel");
                py::object model_instance = OvmsPythonModel();
                py::object kwargs_param = pybind11::dict();
                model_instance.attr("initialize")(kwargs_param);
                this->pythonNodeStates.insert(std::pair<std::string, py::object>(filename, model_instance));
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Failed to process python node file {} : {}", handler_path,  e.what());
                return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
            } catch (...) {
                SPDLOG_ERROR("Failed to process python node file {}", handler_path);
                return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
            }
        }
    }

    return StatusCode::OK;
}

}  // namespace ovms
