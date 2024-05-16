//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include "modelinstance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include <dirent.h>
#include <malloc.h>
#include <openvino/runtime/compiled_model.hpp>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include "capi_frontend/inferencerequest.hpp"
#include "capi_frontend/inferenceresponse.hpp"
#include "config.hpp"
#include "customloaderinterface.hpp"
#include "customloaders.hpp"
#include "deserialization.hpp"
#include "executingstreamidguard.hpp"
#include "filesystem.hpp"
#include "layout.hpp"
#include "layout_configuration.hpp"
#include "logging.hpp"
#include "model_metric_reporter.hpp"
#include "modelconfig.hpp"
#include "modelinstanceunloadguard.hpp"
#include "opencltensorfactory.hpp"
#include "ov_utils.hpp"
#include "predict_request_validation_utils.hpp"
#include "prediction_service_utils.hpp"
#include "profiler.hpp"
#include "serialization.hpp"
#include "shape.hpp"
#include "status.hpp"
#include "stringutils.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"

namespace {
enum : unsigned int {
    GET_INFER_REQUEST,
    PREPROCESS,
    DESERIALIZE,
    PREDICTION,
    SERIALIZE,
    POSTPROCESS,
    TIMER_END
};
}  // namespace

namespace ovms {

const uint MAX_NIREQ_COUNT = 100000;

const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;

ModelInstance::~ModelInstance() = default;
ModelInstance::ModelInstance(const std::string& name, model_version_t version, ov::Core& ieCore, MetricRegistry* registry, const MetricConfig* metricConfig) :
    ieCore(ieCore),
    name(name),
    version(version),
    subscriptionManager(std::string("model: ") + name + std::string(" version: ") + std::to_string(version)),
    status(name, version),
    reporter(std::make_unique<ModelMetricReporter>(metricConfig, registry, name, version)) {
    isCustomLoaderConfigChanged = false;
}

void ModelInstance::subscribe(PipelineDefinition& pd) {
    subscriptionManager.subscribe(pd);
}

void ModelInstance::unsubscribe(PipelineDefinition& pd) {
    subscriptionManager.unsubscribe(pd);
}

static Status getRequestedShape(const ModelConfig& config, const DynamicModelParameter& parameter, const std::string& name, Shape& shapeOut) {
    Shape shape;
    auto mappedName = config.getMappingInputByKey(name);
    auto inputNameToUse = (mappedName != "" ? mappedName : name);
    if (config.getBatchSize().has_value() || parameter.isBatchSizeRequested()) {
        // leave shape untouched
    } else if (config.isShapeAuto(inputNameToUse) && parameter.isShapeRequested(inputNameToUse)) {
        auto status = Shape::fromFlatShape(parameter.getShape(inputNameToUse), shape);
        if (!status.ok()) {
            return status;
        }
    } else if (config.getShapes().count(inputNameToUse) && config.getShapes().at(inputNameToUse).shape.size()) {
        shape = config.getShapes().at(inputNameToUse).shape;
    } else if (config.getShapes().count(ANONYMOUS_INPUT_NAME) && config.getShapes().at(ANONYMOUS_INPUT_NAME).shape.size()) {
        shape = config.getShapes().at(ANONYMOUS_INPUT_NAME).shape;
    }
    shapeOut = shape;
    return StatusCode::OK;
}

static bool hasInputWithName(std::shared_ptr<ov::Model>& model, const std::string& name) {
    try {
        model->input(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

static bool hasOutputWithName(std::shared_ptr<ov::Model>& model, const std::string& name) {
    try {
        model->output(name);
        return true;
    } catch (ov::Exception& e) {
        return false;
    }
}

static Status validateConfigurationAgainstNetwork(const ModelConfig& config, std::shared_ptr<ov::Model>& model) {
    if (config.isShapeAnonymousFixed() && model->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    if (config.getLayout().isSet() && model->inputs().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_LAYOUT_NOT_ALLOWED;
        SPDLOG_LOGGER_WARN(modelmanager_logger, status.string());
        return status;
    }
    for (const auto& [name, _] : config.getShapes()) {
        if (name == ANONYMOUS_INPUT_NAME) {
            continue;
        }
        if (hasInputWithName(model, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(model, name) && !hasInputWithName(model, config.getRealInputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config shape - {} not found in model", name);
            return StatusCode::CONFIG_SHAPE_IS_NOT_IN_MODEL;
        }
    }
    for (const auto& [name, _] : config.getLayouts()) {
        if (hasInputWithName(model, name) && config.getMappingInputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingInputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (hasOutputWithName(model, name) && config.getMappingOutputByKey(name) != "") {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} is mapped by {}. Changes will not apply", name, config.getMappingOutputByKey(name));
            return StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME;
        } else if (!hasInputWithName(model, name) && !hasOutputWithName(model, name) && !hasInputWithName(model, config.getRealInputNameByValue(name)) && !hasOutputWithName(model, config.getRealOutputNameByValue(name))) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Config layout - {} not found in model", name);
            return StatusCode::CONFIG_LAYOUT_IS_NOT_IN_MODEL;
        }
    }
    return StatusCode::OK;
}

const Layout ModelInstance::getReportedTensorLayout(const ModelConfig& config, const std::string& name, bool isInput) {
    Layout defaultLayout;
    if (isInput) {
        OV_LOGGER("ov::Model model: {}, model->input(\"{}\")", reinterpret_cast<const void*>(this->model.get()), name);
        const auto& input = this->model->input(name);
        OV_LOGGER("input: {}, input.get_partial_shape().size()", reinterpret_cast<const void*>(&input));
        defaultLayout = Layout::getDefaultLayout(input.get_partial_shape().size());
        OV_LOGGER("input: {}, input.get_rt_info()", reinterpret_cast<const void*>(&input));
        auto networkSpecifiedLayout = getLayoutFromRTMap(input.get_rt_info());
        if (networkSpecifiedLayout.has_value()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting input layout from RTMap: {}; for tensor name: {}", networkSpecifiedLayout.value().to_string(), name);
            return Layout::fromOvLayout(networkSpecifiedLayout.value());
        }
    } else {
        OV_LOGGER("model: {}, model->output(\"{}\")", reinterpret_cast<const void*>(model.get()), name);
        const auto& output = this->model->output(name);
        OV_LOGGER("output: {}, output.get_partial_shape().size()", reinterpret_cast<const void*>(&output));
        defaultLayout = Layout::getDefaultLayout(output.get_partial_shape().size());
        OV_LOGGER("output: {}, output.get_rt_info()", reinterpret_cast<const void*>(&output));
        auto networkSpecifiedLayout = getLayoutFromRTMap(output.get_rt_info());
        if (networkSpecifiedLayout.has_value()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting output layout from RTMap: {}; for tensor name: {}", networkSpecifiedLayout.value().to_string(), name);
            return Layout::fromOvLayout(networkSpecifiedLayout.value());
        }
    }
    if (isInput && config.getLayout().isSet()) {
        auto layout = config.getLayout().getTensorLayout();
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting layout from ModelConfig: {}; for tensor name: {}", layout, name);
        return layout;
    } else if (config.getLayouts().size() > 0) {
        auto mappedName = config.getMappingInputByKey(name);
        auto it = config.getLayouts().find(mappedName == "" ? name : mappedName);
        if (it != config.getLayouts().end()) {
            auto layout = it->second.getTensorLayout();
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting layout from ModelConfig: {}; for tensor name: {}", layout, name);
            return layout;
        }
    }
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reporting default layout: {}; for tensor name: {}", defaultLayout, name);
    return defaultLayout;
}

static Status applyLayoutConfiguration(const ModelConfig& config, std::shared_ptr<ov::Model>& model, const std::string& modelName, model_version_t modelVersion) {
    OV_LOGGER("ov::Model: {}, ov::preprocess::PrePostProcessor(ov::Model)", reinterpret_cast<void*>(model.get()));
    ov::preprocess::PrePostProcessor preproc(model);

    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Applying layout configuration: {}", config.layoutConfigurationToString());

    OV_LOGGER("ov::Model: {}, model->inputs()", reinterpret_cast<void*>(model.get()));
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        try {
            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_any_name()", reinterpret_cast<const void*>(&input));
            std::string name = input.get_any_name();
            std::string mappedName = config.getMappingInputByKey(name).empty() ? name : config.getMappingInputByKey(name);
            if (config.getLayout().isSet()) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; single input",
                    modelName,
                    modelVersion,
                    config.getLayout().getTensorLayout(),
                    config.getLayout().getModelLayout());

                OV_LOGGER("ov::preprocess::PrePostProcessor::input()::tensor()::set_layout(ov::Layout({}))", config.getLayout().getTensorLayout());
                preproc.input().tensor().set_layout(ov::Layout(config.getLayout().getTensorLayout()));
                OV_LOGGER("ov::preprocess::PrePostProcessor::input()::model()::set_layout(ov::Layout({}))", config.getLayout().getModelLayout());
                preproc.input().model().set_layout(ov::Layout(config.getLayout().getModelLayout()));
            } else if (config.getLayouts().count(mappedName) > 0) {
                auto& layout = config.getLayouts().at(mappedName);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding preprocessing step: Tensor Layout:{}; Network Layout:{}; input name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    mappedName);

                OV_LOGGER("ov::preprocess::PrePostProcessor::input({})::tensor()::set_layout(ov::Layout({}))", name, layout.getTensorLayout());
                preproc.input(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                OV_LOGGER("ov::preprocess::PrePostProcessor::input({})::model()::set_layout(ov::Layout({}))", name, layout.getModelLayout());
                preproc.input(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            } else {
                OV_LOGGER("input: {}, input.get_rt_info()", reinterpret_cast<const void*>(&input));
                auto inheritedModelLayout = getLayoutFromRTMap(input.get_rt_info());
                auto guessedModelLayout = Layout::getDefaultLayout(input.get_partial_shape().size());

                ov::Layout targetModelLayout = inheritedModelLayout.has_value() ? inheritedModelLayout.value() : ov::Layout(guessedModelLayout);

                if (inheritedModelLayout.has_value()) {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (inherited from network); input name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                } else {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (default); input name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                }
                OV_LOGGER("ov::preprocess::PrePostProcessor::input({})::model()::set_layout(ov::Layout({}))", name, targetModelLayout.to_string());
                preproc.input(name).model().set_layout(targetModelLayout);
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure input layout for model:{}; version:{}; from OpenVINO",
                modelName,
                modelVersion);
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    OV_LOGGER("ov::Model: {}, model->outputs()", reinterpret_cast<void*>(model.get()));
    for (const ov::Output<ov::Node>& output : model->outputs()) {
        try {
            OV_LOGGER("ov::Output<ov::Node> output: {}, output.get_any_name()", reinterpret_cast<const void*>(&output));
            std::string name = output.get_any_name();
            std::string mappedName = config.getMappingOutputByKey(name).empty() ? name : config.getMappingOutputByKey(name);
            if (config.getLayouts().count(mappedName) > 0) {
                auto& layout = config.getLayouts().at(mappedName);
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Adding postprocessing step: Tensor Layout:{}; Network Layout:{}; output name: {}",
                    modelName,
                    modelVersion,
                    layout.getTensorLayout(),
                    layout.getModelLayout(),
                    mappedName);
                OV_LOGGER("ov::preprocess::PrePostProcessor::output({})::tensor()::set_layout(ov::Layout({}))", name, layout.getTensorLayout());
                preproc.output(name).tensor().set_layout(ov::Layout(layout.getTensorLayout()));
                OV_LOGGER("ov::preprocess::PrePostProcessor::output({})::model()::set_layout(ov::Layout({}))", name, layout.getModelLayout());
                preproc.output(name).model().set_layout(ov::Layout(layout.getModelLayout()));
            } else {
                OV_LOGGER("output: {}, output.get_rt_info()", reinterpret_cast<const void*>(&output));
                auto inheritedModelLayout = getLayoutFromRTMap(output.get_rt_info());
                auto guessedModelLayout = Layout::getDefaultLayout(output.get_partial_shape().size());

                ov::Layout targetModelLayout = inheritedModelLayout.has_value() ? inheritedModelLayout.value() : ov::Layout(guessedModelLayout);

                if (inheritedModelLayout.has_value()) {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (inherited from network); output name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                } else {
                    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {}, version: {}; Configuring layout: Tensor Layout:; Network Layout:{} (default); output name: {}", modelName, modelVersion, targetModelLayout.to_string(), name);
                }
                OV_LOGGER("ov::preprocess::PrePostProcessor::output({})::model()::set_layout(ov::Layout({}))", name, targetModelLayout.to_string());
                preproc.output(name).model().set_layout(targetModelLayout);
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO with error:{}",
                modelName,
                modelVersion,
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to configure output layout for model:{}; version:{}; from OpenVINO",
                modelName,
                modelVersion);
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    try {
        OV_LOGGER("preproc: {}, ov::Model = ov::preprocess::PrePostProcessor::build()", reinterpret_cast<void*>(&preproc));
        model = preproc.build();
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot change layout");
        return StatusCode::MODEL_NOT_LOADED;
    }
    return StatusCode::OK;
}

const std::string RT_INFO_KEY{"model_info"};

ov::AnyMap ModelInstance::getRTInfo() const {
    OV_LOGGER("model: {}, ov::Model::has_rt_info({})", reinterpret_cast<void*>(model.get()), RT_INFO_KEY);
    if (this->model->has_rt_info(RT_INFO_KEY)) {
        OV_LOGGER("model: {}, ov::Model::get_rt_info<ov::AnyMap>({})", reinterpret_cast<void*>(model.get()), RT_INFO_KEY);
        return model->get_rt_info<ov::AnyMap>(RT_INFO_KEY);
    }
    OV_LOGGER("ov::AnyMap()");
    return ov::AnyMap();
}

Status ModelInstance::loadTensors(const ModelConfig& config, bool needsToApplyLayoutConfiguration, const DynamicModelParameter& parameter) {
    Status status = validateConfigurationAgainstNetwork(config, this->model);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during configuration validation against model");
        return status;
    }
    if (needsToApplyLayoutConfiguration) {
        status = applyLayoutConfiguration(config, this->model, getName(), getVersion());
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during layout configuration");
            return status;
        }
    }
    status = loadInputTensors(config, parameter);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during loading input tensors");
        return status;
    }
    status = loadOutputTensors(config);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during loading output tensors");
        return status;
    }
    return StatusCode::OK;
}

Status ModelInstance::gatherReshapeInfo(bool isBatchingModeAuto, const DynamicModelParameter& parameter, bool& isReshapeRequired, std::map<std::string, ov::PartialShape>& modelShapes) {
    OV_LOGGER("ov::Model: {}, model->inputs()", reinterpret_cast<void*>(model.get()));
    for (const ov::Output<ov::Node>& input : this->model->inputs()) {
        std::string name;
        try {
            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_any_name()", reinterpret_cast<const void*>(&input));
            std::string name = input.get_any_name();
            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_partial_shape()", reinterpret_cast<const void*>(&input));
            ov::PartialShape shape = input.get_partial_shape();
            if (shape.size() == 0 && isBatchingModeAuto) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to load model:{}; version:{}; with batching=AUTO due to existing scalar input name:{}",
                    getName(), getVersion(), name);
                return StatusCode::MODEL_WITH_SCALAR_AUTO_UNSUPPORTED;
            }
            Shape requestedShape;
            auto status = getRequestedShape(config, parameter, name, requestedShape);
            if (!status.ok()) {
                return status;
            }
            if (requestedShape.size() > 0) {
                shape = requestedShape.createPartialShape();
            }
            modelShapes[name] = shape;
            if (input.get_partial_shape() != shape) {
                isReshapeRequired = true;
            }
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    return StatusCode::OK;
}

Status ModelInstance::loadInputTensors(const ModelConfig& config, const DynamicModelParameter& parameter) {
    this->inputsInfo.clear();

    std::map<std::string, ov::PartialShape> modelShapes;
    bool reshapeRequired = false;

    bool isBatchingModeAuto = config.getBatchingMode() == Mode::AUTO;

    auto status = gatherReshapeInfo(isBatchingModeAuto, parameter, reshapeRequired, modelShapes);
    if (!status.ok()) {
        return status;
    }

    if (reshapeRequired) {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs", getName(), getVersion());
        try {
            OV_LOGGER("ov::Model: {}, model->reshape(modelShapes)", reinterpret_cast<void*>(model.get()));
            model->reshape(modelShapes);
        } catch (const ov::Exception& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        }
    } else {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs is not required", getName(), getVersion());
    }
    configureBatchSize(this->config, parameter);

    OV_LOGGER("ov::Model: {}, model->inputs()", reinterpret_cast<void*>(model.get()));
    for (const ov::Output<ov::Node>& input : this->model->inputs()) {
        try {
            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_any_name()", reinterpret_cast<const void*>(&input));
            std::string name = input.get_any_name();

            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_element_type()", reinterpret_cast<const void*>(&input));
            ovms::Precision precision = ovElementTypeToOvmsPrecision(input.get_element_type());
            OV_LOGGER("ov::Output<ov::Node> input: {}, input.get_partial_shape()", reinterpret_cast<const void*>(&input));
            Shape shape(input.get_partial_shape());
            std::string mappingName = config.getMappingInputByKey(name);
            const Layout layout = getReportedTensorLayout(config, name, true);

            if (!layout.isCompatible(shape)) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Layout: {}; incompatible with shape: {}; for input name: {}", layout, shape.toString(), name);
                return StatusCode::LAYOUT_INCOMPATIBLE_WITH_SHAPE;
            }

            std::shared_ptr<const TensorInfo> info = std::make_shared<TensorInfo>(
                name,
                mappingName,
                precision,
                shape,
                layout);

            SPDLOG_LOGGER_INFO(modelmanager_logger, "Input {}", info->asString());
            this->inputsInfo[info->getMappedName()] = std::move(info);
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get input name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOutputTensors(const ModelConfig& config) {
    this->outputsInfo.clear();

    OV_LOGGER("ov::Model model: {}, model->outputs()", reinterpret_cast<void*>(model.get()));
    for (const ov::Output<ov::Node>& output : this->model->outputs()) {
        try {
            OV_LOGGER("ov::Output<ov::Node> output: {}, output.get_any_name()", reinterpret_cast<const void*>(&output));
            std::string name = output.get_any_name();

            OV_LOGGER("ov::Output<ov::Node> output: {}, output.get_element_type()", reinterpret_cast<const void*>(&output));
            ovms::Precision precision = ovElementTypeToOvmsPrecision(output.get_element_type());
            OV_LOGGER("ov::Output<ov::Node> output: {}, output.get_partial_shape()", reinterpret_cast<const void*>(&output));
            Shape shape(output.get_partial_shape());
            std::string mappingName = config.getMappingOutputByKey(name);
            const Layout layout = getReportedTensorLayout(config, name, false);

            if (!layout.isCompatible(shape)) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Layout: {}; incompatible with shape: {}; for output name: {}", layout, shape.toString(), name);
                return StatusCode::LAYOUT_INCOMPATIBLE_WITH_SHAPE;
            }

            std::shared_ptr<const TensorInfo> info = std::make_shared<TensorInfo>(
                name,
                mappingName,
                precision,
                shape,
                layout);

            SPDLOG_LOGGER_INFO(modelmanager_logger, "Output {}", info->asString());

            this->outputsInfo[info->getMappedName()] = std::move(info);
        } catch (const ov::Exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO with error:{}",
                getName(),
                getVersion(),
                e.what());
            return StatusCode::UNKNOWN_ERROR;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get output name for model:{}; version:{}; from OpenVINO",
                getName(),
                getVersion());
            return StatusCode::UNKNOWN_ERROR;
        }
    }

    return StatusCode::OK;
}

// Temporary methods. To be replaces with proper storage class.
static bool dirExists(const std::string& path) {
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return false;
    }
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }

    return false;
}

static std::string findFilePathWithExtension(const std::string& path, const std::string& extension) {
    struct dirent* entry;
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return std::string();
    }
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        SPDLOG_WARN("Failed to opendir: {}", path);
        return std::string();
    }

    while ((entry = readdir(dir)) != nullptr) {
        auto name = std::string(entry->d_name);
        if (endsWith(name, extension)) {
            closedir(dir);
            if (endsWith(name, "/")) {
                return path + name;
            } else {
                return path + '/' + name;
            }
        }
    }
    closedir(dir);

    return std::string();
}

std::string ModelInstance::findModelFilePathWithExtension(const std::string& extension) const {
    return findFilePathWithExtension(path, extension);
}

uint ModelInstance::getNumOfParallelInferRequestsUnbounded(const ModelConfig& modelConfig) {
    uint numberOfParallelInferRequests = 0;
    if (modelConfig.getNireq() > 0) {
        return modelConfig.getNireq();
    }
    auto& ovmsConfig = ovms::Config::instance();
    if (ovmsConfig.nireq() > 0) {
        // nireq is set globally for all models in ovms startup parameters
        return ovmsConfig.nireq();
    }
    try {
        numberOfParallelInferRequests = compiledModel->get_property(ov::optimal_number_of_infer_requests);
    } catch (const ov::Exception& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    return numberOfParallelInferRequests;
}

uint ModelInstance::getNumOfParallelInferRequests(const ModelConfig& modelConfig) {
    uint nireq = getNumOfParallelInferRequestsUnbounded(modelConfig);
    if (nireq > MAX_NIREQ_COUNT) {
        SPDLOG_WARN("Invalid nireq because its value was too high: {}. Maximum value: {}", nireq, MAX_NIREQ_COUNT);
        return 0;
    } else if (nireq < 1u) {
        SPDLOG_WARN("Ignored configured nireq because it has to be above 0 and was: {}. Set to 1", nireq);
        return 1u;
    }
    return nireq;
}

std::shared_ptr<ov::Model> ModelInstance::loadOVModelPtr(const std::string& modelFile) {
    OV_LOGGER("ov::Core: {}, model = ieCore.read_model(\"{}\")", reinterpret_cast<const void*>(&this->ieCore), modelFile);
    return this->ieCore.read_model(modelFile);
}

Status ModelInstance::loadOVModel() {
    auto& modelFile = modelFiles[0];
    SPDLOG_DEBUG("Try reading model file: {}", modelFile);
    try {
        this->model = loadOVModelPtr(modelFile);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOVModelUsingCustomLoader() {
    SPDLOG_DEBUG("Try reading model using a custom loader");
    try {
        std::vector<uint8_t> modelBinary;
        std::vector<uint8_t> weights;

        SPDLOG_INFO("loading ov::Model for model: {} basepath: {} <> {} version: {}", getName(), getPath(), this->config.getBasePath().c_str(), getVersion());

        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];

        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_INFO("Loader {} is not in loaded customloaders list", loaderName);
            throw std::invalid_argument("customloader not exisiting");
        }

        CustomLoaderStatus res = customLoaderInterfacePtr->loadModel(this->config.getName(),
            this->config.getBasePath(),
            getVersion(),
            this->config.getCustomLoaderOptionsConfigStr(), modelBinary, weights);

        if (res == CustomLoaderStatus::MODEL_LOAD_ERROR) {
            return StatusCode::FILE_INVALID;
        }

        if ((res == CustomLoaderStatus::INTERNAL_ERROR) || (res == CustomLoaderStatus::MODEL_BLACKLISTED)) {
            return StatusCode::INTERNAL_ERROR;
        }

        std::string strModel(modelBinary.begin(), modelBinary.end());

        if (res == CustomLoaderStatus::MODEL_TYPE_IR) {
            ov::Tensor tensorWts(ov::element::u8, ov::Shape{weights.size()});
            std::memcpy(tensorWts.data(), weights.data(), weights.size());
            model = ieCore.read_model(strModel, tensorWts);
        } else if (res == CustomLoaderStatus::MODEL_TYPE_ONNX) {
            model = ieCore.read_model(strModel, ov::Tensor());
        } else if (res == CustomLoaderStatus::MODEL_TYPE_BLOB) {
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (ov::Exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading ov::Model for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}
}  // namespace ovms
//#include <CL/cl.h>
//#include "openvino/runtime/intel_gpu/properties.hpp"
//#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#include "ocl_utils.hpp"
#include "openvino/runtime/remote_tensor.hpp"
namespace ovms {
void ModelInstance::loadCompiledModelPtr(const plugin_config_t& pluginConfig) {
    OV_LOGGER("ov::Core: {}, ov::Model: {}, targetDevice: {}, ieCore.compile_model(model, targetDevice, pluginConfig", reinterpret_cast<void*>(&ieCore), reinterpret_cast<void*>(this->model.get()), this->targetDevice);
    cl_platform_id platformId;
    cl_device_id deviceId;
    //auto ocl_context = getOCLContext();  // TODO use params from config, use device from config
    auto ocl_context = get_cl_context(platformId, deviceId);  // TODO use params from config, use device from config
    SPDLOG_ERROR("XXXXXXXX Loading model with context");
    this->ocl_context = ocl_context;
    //   auto ov_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    this->ocl_context_cpp = std::make_unique<ov::intel_gpu::ocl::ClContext>(ieCore, ocl_context);
    ov::intel_gpu::ocl::ClContext ov_context(ieCore, ocl_context);
    //compiledModel = std::make_shared<ov::CompiledModel>(ieCore.compile_model(this->model, this->targetDevice, pluginConfig));
    compiledModel = std::make_shared<ov::CompiledModel>(ieCore.compile_model(this->model, *this->ocl_context_cpp, pluginConfig));
}

plugin_config_t ModelInstance::prepareDefaultPluginConfig(const ModelConfig& config) {
    plugin_config_t pluginConfig = config.getPluginConfig();
    if ((pluginConfig.count("NUM_STREAMS") == 1) || (pluginConfig.count("PERFORMANCE_HINT") == 1)) {
        return pluginConfig;
    } else {
        pluginConfig["PERFORMANCE_HINT"] = "LATENCY";
    }
    return pluginConfig;
}

Status ModelInstance::loadOVCompiledModel(const ModelConfig& config) {
    plugin_config_t pluginConfig = prepareDefaultPluginConfig(config);
    try {
        loadCompiledModelPtr(pluginConfig);
    } catch (ov::Exception& e) {
        Status status = StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    } catch (std::exception& e) {
        Status status = StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE;
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
    }

    uint32_t numberOfStreams = getNumOfStreams();
    SET_IF_ENABLED(getMetricReporter().streams, numberOfStreams);

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Plugin config for device: {}", targetDevice);
    for (const auto& pair : pluginConfig) {
        const auto& key = pair.first;
        const auto& value = pair.second;
        SPDLOG_LOGGER_INFO(modelmanager_logger, "OVMS set plugin settings key: {}; value: {};", key, value.as<std::string>());
    }
    logOVPluginConfig([this](const std::string& key) {
            OV_LOGGER("ov::CompiledModel:{} get_property({})", reinterpret_cast<void*>(this->model.get()), key);
            return this->compiledModel->get_property(key); },
        std::string("compiled model: ") + getName(),
        std::string(" version: ") + std::to_string(getVersion()) + std::string("; target device: ") + targetDevice + ";");
    return StatusCode::OK;
}

template <class ArrayType>
void ModelInstance::fetchModelFiles(bool& found, ArrayType ext) {
    if (!found) {
        found = true;
        modelFiles.clear();
        for (auto extension : ext) {
            auto file = findModelFilePathWithExtension(extension);
            if (file.empty()) {
                found = false;
            }
            if (endsWith(file, "saved_model.pb")) {
                file = file.substr(0, file.find("saved_model.pb"));
            }
            modelFiles.push_back(file);
        }
    }
}

Status ModelInstance::fetchModelFilepaths() {
    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        // not required if the model is loaded using a custom loader and can be returned from here
        return StatusCode::OK;
    }

    SPDLOG_DEBUG("Getting model files from path: {}", path);
    if (!dirExists(path)) {
        SPDLOG_ERROR("Missing model directory {}", path);
        return StatusCode::PATH_INVALID;
    }

    bool found = false;
    fetchModelFiles(found, OV_MODEL_FILES_EXTENSIONS);
    fetchModelFiles(found, ONNX_MODEL_FILES_EXTENSIONS);
    fetchModelFiles(found, PADDLE_MODEL_FILES_EXTENSIONS);
    fetchModelFiles(found, TF_MODEL_FILES_EXTENSIONS);
    fetchModelFiles(found, TFLITE_MODEL_FILES_EXTENSIONS);

    if (!found) {
        SPDLOG_ERROR("Could not find file for model: {} version: {} in path: {}", getName(), getVersion(), path);
        return StatusCode::FILE_INVALID;
    }

    return StatusCode::OK;
}

Status ModelInstance::prepareInferenceRequestsQueue(const ModelConfig& config) {
    uint numberOfParallelInferRequests = getNumOfParallelInferRequests(config);
    if (numberOfParallelInferRequests == 0) {
        return Status(StatusCode::INVALID_NIREQ, "Exceeded allowed nireq value");
    }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(*compiledModel, numberOfParallelInferRequests);
    SET_IF_ENABLED(this->getMetricReporter().inferReqQueueSize, numberOfParallelInferRequests);
    auto batchSize = getBatchSize();
    SPDLOG_INFO("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
        getName(),
        getVersion(),
        batchSize.has_value() ? batchSize.value().toString() : std::string{"none"},
        numberOfParallelInferRequests);
    return StatusCode::OK;
}

void ModelInstance::configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter) {
    if (parameter.isBatchSizeRequested()) {
        OV_LOGGER("ov::Model: {}, ov::set_batch({})", reinterpret_cast<void*>(this->model.get()), parameter.getBatchSize());
        ov::set_batch(model, parameter.getBatchSize());
    } else if (config.getBatchSize().has_value()) {
        OV_LOGGER("ov::Model: {}, ov::set_batch({})", reinterpret_cast<void*>(this->model.get()), ovms::Dimension(config.getBatchSize().value().createPartialDimension()).toString());
        ov::set_batch(model, config.getBatchSize().value().createPartialDimension());
    }
}

Status ModelInstance::loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter) {
    bool isLayoutConfigurationChanged = !config.isLayoutConfigurationEqual(this->config);
    bool needsToApplyLayoutConfiguration = isLayoutConfigurationChanged || !this->model;

    subscriptionManager.notifySubscribers();
    this->path = config.getPath();
    this->targetDevice = config.getTargetDevice();
    this->config = config;
    auto status = fetchModelFilepaths();

    if (!status.ok()) {
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return status;
    }
    try {
        status = setCacheOptions(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        if (!this->model || isLayoutConfigurationChanged) {
            if (this->config.isCustomLoaderRequiredToLoadModel()) {
                status = loadOVModelUsingCustomLoader();
            } else {
                status = loadOVModel();
            }
        }

        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        status = loadTensors(this->config, needsToApplyLayoutConfiguration, parameter);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = loadOVCompiledModel(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = prepareInferenceRequestsQueue(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
    } catch (const ov::Exception& e) {
        SPDLOG_ERROR("exception occurred while loading model: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::MODEL_NOT_LOADED;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("exception occurred while loading model: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::MODEL_NOT_LOADED;
    }
    try {
        OV_LOGGER("ov::CompiledModel: {} compiledModel->get_property(ov::loaded_from_cache)", reinterpret_cast<void*>(compiledModel.get()));
        bool isModelLoadedFromCache = compiledModel->get_property(ov::loaded_from_cache);
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Is model loaded from cache: {}", isModelLoadedFromCache);
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Unable to get information if model was loaded from cache; model: {}; version: {}; device: {}", getName(), getVersion(), config.getTargetDevice());
    }
    this->status.setAvailable();
    modelLoadedNotify.notify_all();
    return status;
}

Status ModelInstance::setCacheOptions(const ModelConfig& config) {
    if (!config.getCacheDir().empty()) {
        if (!config.isAllowCacheSetToTrue() && (config.isCustomLoaderRequiredToLoadModel() || config.anyShapeSetToAuto() || (config.getBatchingMode() == Mode::AUTO))) {
            OV_LOGGER("ov::Core: {}, ieCore.set_property(ov::cache_dir({}))", reinterpret_cast<const void*>(this->compiledModel.get()), "");
            this->ieCore.set_property(ov::cache_dir(""));
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has disabled caching", this->getName());
            this->cacheDisabled = true;
        } else if (config.isAllowCacheSetToTrue() && config.isCustomLoaderRequiredToLoadModel()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model: {} has allow cache set to true while using custom loader", this->getName());
            return StatusCode::ALLOW_CACHE_WITH_CUSTOM_LOADER;
        } else {
            OV_LOGGER("ov::Core: {}, ieCore.set_property(ov::cache_dir({}))", reinterpret_cast<const void*>(this->compiledModel.get()), config.getCacheDir());
            this->ieCore.set_property(ov::cache_dir(config.getCacheDir()));
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Model: {} has enabled caching", this->getName());
        }
    }
    return StatusCode::OK;
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Loading model: {}, version: {}, from path: {}, with target device: {} ...",
        config.getName(), config.getVersion(), config.getPath(), config.getTargetDevice());
    if (config.getBatchingMode() == AUTO) {
        SPDLOG_INFO("Batch size mode for model {} is set to auto", config.getName());
    } else if (config.anyShapeSetToAuto()) {
        SPDLOG_INFO("Some inputs shapes for model {} are set to auto", config.getName());
    }
    this->status = ModelVersionStatus(config.getName(), config.getVersion());
    this->status.setLoading();
    return loadModelImpl(config);
}

Status ModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to reload model: {} version: {}. Blocked by: {} inferences in progress.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    if ((this->config.isCustomLoaderRequiredToLoadModel()) && (isCustomLoaderConfigChanged)) {
        // unloading and the loading back the model
        isCustomLoaderConfigChanged = false;
        retireModel(isCustomLoaderConfigChanged);
    }
    return loadModelImpl(config, parameter);
}

Status ModelInstance::recoverFromReloadingError(const Status& status) {
    SPDLOG_WARN("Failed to perform complete reload with requested dynamic parameter. Model: {} version: {} with error: {}. Reloading to previous configuration",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to recover model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return status;
}

Status ModelInstance::reshapeWithFullReload(const Status& status, const DynamicModelParameter& parameter) {
    SPDLOG_WARN("Failed to reload model: {} version: {} with error: {}. Trying to perform complete reload with requested dynamic parameter",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config, parameter);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to reload model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return recoveryStatus;
}

Status ModelInstance::reloadModel(std::optional<Dimension> batchSize, std::map<std::string, shape_t> requestShapes, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuard) {
    // temporarily release current predictRequest lock on model loading
    unloadGuard.reset();
    // block concurrent requests for reloading/unloading - assure that after reload predict request
    // will block further requests for reloading/unloading until inference is performed
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Will reload model: {} version: {}", getName(), getVersion());

    DynamicModelParameter parameter;
    if (batchSize.has_value() && batchSize.value().isStatic()) {
        parameter = DynamicModelParameter(batchSize.value().getStaticValue());
    } else if (requestShapes.size() > 0) {
        parameter = DynamicModelParameter(requestShapes);
    } else {
        SPDLOG_DEBUG("Error: requested model: {} version: {} reload with no batchsize and shapes set.", getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }

    auto status = reloadModel(config, parameter);
    if (!status.ok()) {
        status = this->reshapeWithFullReload(status, parameter);
        if (!status.ok()) {
            return this->recoverFromReloadingError(status);
        }
    }
    unloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    return status;
}

Status ModelInstance::reloadModelIfRequired(
    Status validationStatus,
    const std::optional<Dimension>& requestedBatchSize,
    const std::map<std::string, shape_t>& requestedShapes,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Status status = std::move(validationStatus);
    if (status.batchSizeChangeRequired()) {
        try {
            status = reloadModel(requestedBatchSize, {}, modelUnloadGuardPtr);
        } catch (const std::exception& e) {
            status = Status(StatusCode::INVALID_BATCH_DIMENSION, e.what());
        }
        if (!status.ok()) {
            SPDLOG_ERROR("Model: {}, version: {} reload (batch size change) failed. Status Code: {}, Error {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (status.reshapeRequired()) {
        status = reloadModel(std::nullopt, requestedShapes, modelUnloadGuardPtr);
        if (!status.ok() && status != StatusCode::RESHAPE_ERROR) {
            SPDLOG_ERROR("Model: {}, version: {} reload (reshape) failed. Status Code: {}, Error: {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (!status.ok()) {
        SPDLOG_DEBUG("Model: {}, version: {} validation of inferRequest failed. Status Code: {}, Error: {}",
            getName(), getVersion(), status.getCode(), status.string());
    }
    return status;
}

Status ModelInstance::waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuard) {
    // order is important here for performance reasons
    // assumption: model is already loaded for most of the calls
    modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    if (getStatus().getState() == ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Model: {}, version: {} already loaded", getName(), getVersion());
        return StatusCode::OK;
    }
    modelInstanceUnloadGuard.reset();

    // wait several time since no guarantee that cv wakeup will be triggered before calling wait_for
    const uint waitLoadedTimestepMilliseconds = 100;
    const uint waitCheckpoints = waitForModelLoadedTimeoutMilliseconds / waitLoadedTimestepMilliseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    SPDLOG_DEBUG("Waiting for loaded state for model: {} version: {} with timestep: {} timeout: {} check count: {}", getName(), getVersion(),
        waitLoadedTimestepMilliseconds, waitForModelLoadedTimeoutMilliseconds, waitCheckpointsCounter);
    std::mutex cv_mtx;
    std::unique_lock<std::mutex> cv_lock(cv_mtx);
    while (waitCheckpointsCounter-- > 0) {
        if (modelLoadedNotify.wait_for(cv_lock,
                std::chrono::milliseconds(waitLoadedTimestepMilliseconds),
                [this]() {
                    return this->getStatus().getState() > ModelVersionState::LOADING;
                })) {
            SPDLOG_INFO("Waiting for model: {} version: {} loaded state for: {} time",
                getName(), getVersion(), waitCheckpoints - waitCheckpointsCounter);
        }
        modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
        if (getStatus().getState() == ModelVersionState::AVAILABLE) {
            SPDLOG_INFO("Succesfully waited for model: {}, version: {}", getName(), getVersion());
            return StatusCode::OK;
        }
        modelInstanceUnloadGuard.reset();
        if (ModelVersionState::AVAILABLE < getStatus().getState()) {
            SPDLOG_INFO("Stopped waiting for model: {} version: {} since it is unloading.", getName(), getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_INFO("Waiting for loaded state reached timeout for model: {} version: {}",
        getName(), getVersion());
    if (getStatus().getState() > ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended since it started unloading.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    } else {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended due to timeout.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
}

void ModelInstance::retireModel(bool isPermanent) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    if (isPermanent) {
        this->status.setUnloading();
    } else {
        this->status.setLoading();
    }
    unloadModelComponents();
    if (isPermanent) {
        status.setEnd();
    }
}

void ModelInstance::cleanupFailedLoad() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
    unloadModelComponents();
}

void ModelInstance::unloadModelComponents() {
    subscriptionManager.notifySubscribers();
    while (!canUnloadInstance()) {
        SPDLOG_DEBUG("Waiting to unload model: {} version: {}. Blocked by: {} inferences in progres.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    SET_IF_ENABLED(this->getMetricReporter().inferReqQueueSize, 0);
    SET_IF_ENABLED(this->getMetricReporter().streams, 0);
    inferRequestsQueue.reset();
    compiledModel.reset();
    model.reset();
    outputsInfo.clear();
    inputsInfo.clear();
    modelFiles.clear();

    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];
        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "The loader {} is no longer available for model: {} version : {}",
                loaderName, getName(), getVersion());
        } else {
            // once model is unloaded, notify custom loader object about the unload
            customLoaderInterfacePtr->unloadModel(getName(), getVersion());
        }
    }
    malloc_trim(0);
}

const std::set<std::string>& ModelInstance::getOptionalInputNames() {
    static const std::set<std::string> optionalInputNames = {};
    return optionalInputNames;
}

template <typename RequestType>
const Status ModelInstance::validate(const RequestType* request) {
    OVMS_PROFILE_FUNCTION();
    return request_validation_utils::validate(
        *request,
        getInputsInfo(),
        getName(),
        getVersion(),
        this->getOptionalInputNames(),
        getModelConfig().getBatchingMode(),
        getModelConfig().getShapes());
}

template const Status ModelInstance::validate(const InferenceRequest* request);
template const Status ModelInstance::validate(const ::KFSRequest* request);
template const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request);

Status ModelInstance::performInference(ov::InferRequest& inferRequest) {
    OVMS_PROFILE_FUNCTION();
    try {
        enum : unsigned int {
            INFER,
            TIMER_END2
        };
        Timer<TIMER_END2> timer;
        timer.start(INFER);
        OV_LOGGER("ov::InferRequest: {}, inferRequest.start_async()", reinterpret_cast<void*>(&inferRequest));
        OVMS_PROFILE_SYNC_BEGIN("ov::InferRequest::start_async");
        inferRequest.start_async();
        OVMS_PROFILE_SYNC_END("ov::InferRequest::start_async");
        OV_LOGGER("ov::InferRequest: {}, inferRequest.wait()", reinterpret_cast<void*>(&inferRequest));
        OVMS_PROFILE_SYNC_BEGIN("ov::InferRequest::wait");
        inferRequest.wait();
        OVMS_PROFILE_SYNC_END("ov::InferRequest::wait");
        timer.stop(INFER);
        double inferTime = timer.elapsed<std::chrono::microseconds>(INFER);
        OBSERVE_IF_ENABLED(this->getMetricReporter().inferenceTime, inferTime);
    } catch (const ov::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async caught an exception {}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

template <typename RequestType, typename ResponseType>
Status ModelInstance::infer(const RequestType* requestProto,
    ResponseType* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;

    auto requestProcessor = createRequestProcessor(requestProto, responseProto);  // request, response passed only to deduce type
    auto status = requestProcessor->extractRequestParameters(requestProto);
    if (!status.ok())
        return status;
    status = validate(requestProto);
    if (status.batchSizeChangeRequired() || status.reshapeRequired()) {
        // We are ensured that request shape is valid and convertible to model shape (non negative, non zero)
        // We can use it to perform reshape via shape=auto
        auto requestBatchSize = getRequestBatchSize(requestProto, this->getBatchSizeIndex());
        auto requestShapes = getRequestShapes(requestProto);
        status = reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    }
    if (!status.ok())
        return status;
    status = requestProcessor->prepare();
    if (!status.ok())
        return status;

    timer.start(GET_INFER_REQUEST);
    OVMS_PROFILE_SYNC_BEGIN("getInferRequest");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue(), this->getMetricReporter());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    OVMS_PROFILE_SYNC_END("getInferRequest");
    timer.stop(GET_INFER_REQUEST);
    double getInferRequestTime = timer.elapsed<microseconds>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(this->getMetricReporter().waitForInferReqTime, getInferRequestTime);
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, getInferRequestTime / 1000);

    timer.start(PREPROCESS);
    status = requestProcessor->preInferenceProcessing(inferRequest);
    timer.stop(PREPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(PREPROCESS) / 1000);

    timer.start(DESERIALIZE);
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    ov::intel_gpu::ocl::ClContext ovOclContext(this->ieCore, this->ocl_context);
    OpenCLTensorFactory factory(*this->ocl_context_cpp);
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inputSink, isPipeline, &factory);
    // TODO FIXME check status
    isPipeline = true;
    status = deserializePredictRequest2<ConcreteTensorProtoDeserializator, InputSink<ov::InferRequest&>, true>(*requestProto, getOutputsInfo(), inputSink, isPipeline, &factory);
    timer.stop(DESERIALIZE);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(DESERIALIZE) / 1000);

    timer.start(PREDICTION);
    status = performInference(inferRequest);
    timer.stop(PREDICTION);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(PREDICTION) / 1000);

    timer.start(SERIALIZE);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    status = serializePredictResponse(outputGetter, getName(), getVersion(), getOutputsInfo(), requestProto, responseProto, getTensorInfoName, useSharedOutputContentFn(requestProto));
    timer.stop(SERIALIZE);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(SERIALIZE) / 1000);

    timer.start(POSTPROCESS);
    status = requestProcessor->postInferenceProcessing(responseProto, inferRequest);
    timer.stop(POSTPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Postprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(POSTPROCESS) / 1000);
    if (this->targetDevice == "AUTO")
        for (std::string device : compiledModel->get_property(ov::execution_devices))
            SPDLOG_DEBUG("Used device: {}", device);

    status = requestProcessor->release();
    return status;
}
template <typename RequestType>
void splitOutputs(const tensor_map_t& outputsInfo, const RequestType& request, std::set<std::string>& outputsInRequest, std::set<std::string>& outputsNotInRequest){};

template <>
void splitOutputs<InferenceRequest>(const tensor_map_t& outputsInfo, const InferenceRequest& request, std::set<std::string>& outputsInRequest, std::set<std::string>& outputsNotInRequest) {
    for (auto& [name, info] : outputsInfo) {
        const InferenceTensor* tensor{nullptr};
        auto status = request.getOutput(name.c_str(), &tensor);
        if (status.ok() && tensor != nullptr) {
            outputsInRequest.emplace(name);
        } else {
            outputsNotInRequest.emplace(name);
        }
    }
}
template <typename RequestType, typename ResponseType>
Status ModelInstance::inferAsync(const RequestType* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;
    // TODO FIXME we don't have response yet!
    //auto requestProcessor = createRequestProcessor(requestProto, responseProto);  // request, response passed only to deduce type
    //auto status = requestProcessor->extractRequestParameters(requestProto);
    //if (!status.ok())
    //    return status;
    auto status = validate(requestProto);
    if (status.batchSizeChangeRequired() || status.reshapeRequired()) {
        // We are ensured that request shape is valid and convertible to model shape (non negative, non zero)
        // We can use it to perform reshape via shape=auto
        auto requestBatchSize = getRequestBatchSize(requestProto, this->getBatchSizeIndex());
        auto requestShapes = getRequestShapes(requestProto);
        status = reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    }
    if (!status.ok())
        return status;
    /*status = requestProcessor->prepare();
    if (!status.ok())
        return status;
*/
    timer.start(GET_INFER_REQUEST);
    OVMS_PROFILE_SYNC_BEGIN("getInferRequest");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue(), this->getMetricReporter());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    OVMS_PROFILE_SYNC_END("getInferRequest");
    timer.stop(GET_INFER_REQUEST);
    double getInferRequestTime = timer.elapsed<microseconds>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(this->getMetricReporter().waitForInferReqTime, getInferRequestTime);
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, getInferRequestTime / 1000);

    /*timer.start(PREPROCESS);
    status = requestProcessor->preInferenceProcessing(inferRequest);
    timer.stop(PREPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(PREPROCESS) / 1000);
*/
    timer.start(DESERIALIZE);
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    ov::intel_gpu::ocl::ClContext ovOclContext(this->ieCore, this->ocl_context);
    OpenCLTensorFactory factory(*this->ocl_context_cpp);
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inputSink, isPipeline, &factory);
    if (!status.ok()) {
        SPDLOG_DEBUG("Deserialization of inputs failed for model {}, version {}",
            getName(), getVersion());
        return status;
    }
    status = deserializePredictRequest2<ConcreteTensorProtoDeserializator, InputSink<ov::InferRequest&>, true>(*requestProto, getOutputsInfo(), inputSink, isPipeline, &factory);
    timer.stop(DESERIALIZE);
    if (!status.ok()) {
        SPDLOG_DEBUG("Deserialization of outputs failed for model {}, version {}", getName(), getVersion());
        return status;
    }
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        getName(), getVersion(), executingInferId, timer.elapsed<microseconds>(DESERIALIZE) / 1000);
    // set callback
    OVMS_InferenceResponseCompleteCallback_t userCallback = requestProto->getResponseCompleteCallback();
    void* userCallbackData = requestProto->getResponseCompleteCallbackData();
    // here pass by copy into callback
    // TODO unload model guard & test
    inferRequest.set_callback([this, requestProto, &inferRequest, userCallback, userCallbackData](std::exception_ptr exception) {
        SPDLOG_INFO("Entry of ov::InferRequest callback call");
        if (exception) {
            try {
                std::rethrow_exception(exception);
            } catch (const std::exception& e) {
                SPDLOG_DEBUG("got exception in ov::InferRequest callback: {}", e.what());
            } catch (...) {
                SPDLOG_DEBUG("got exception in ov::InferRequest callback");
                return;
            }
        }
        SPDLOG_INFO("Using OV callback");
        // here use OVMS response serialization
        // here call user set callback
        // here will go OVMS C-API serialization code
        std::unique_ptr<ovms::InferenceResponse> res(new ovms::InferenceResponse(this->getName(), this->getVersion()));
        OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
        try {
            // TODO created filter based on what is in request, then perform casual serialization for what was NOT in request, and rewrite tensors from request to response for those that were
            std::set<std::string> outputsInRequest;
            std::set<std::string> outputsNotInRequest;
            splitOutputs(getOutputsInfo(), requestProto, outputsInRequest, outputsNotInRequest);
            auto status = serializePredictResponse(outputGetter, getName(), getVersion(), getOutputsInfo(), requestProto, res.get(), getTensorInfoName, useSharedOutputContentFn(requestProto));  // TODO FIXME handle status
        } catch (std::exception& e) {
            SPDLOG_DEBUG("caught exception in ov::InferRequest callback: {}", e.what());
        } catch (...) {
            SPDLOG_DEBUG("caught exception in ov::InferRequest callback");
        }
        OVMS_InferenceResponse* response = reinterpret_cast<OVMS_InferenceResponse*>(res.release());
        SPDLOG_DEBUG("Calling user provided callback");  // TODO check if this shows
        userCallback(response, 1, userCallbackData);
        SPDLOG_DEBUG("Called user provided callback");                       // TODO check if this shows
        inferRequest.set_callback([](std::exception_ptr exception_ptr) {});  // reset callback on infer request
    });
    OV_LOGGER("ov::InferRequest: {}, inferRequest.start_async()", reinterpret_cast<void*>(&inferRequest));
    inferRequest.start_async();  // TODO wrap into try catch
    return StatusCode::OK;
}

template Status ModelInstance::infer<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>(const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

template Status ModelInstance::inferAsync<InferenceRequest, InferenceResponse>(const InferenceRequest* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

template Status ModelInstance::infer(const ::KFSRequest* requestProto,
    ::KFSResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

const size_t ModelInstance::getBatchSizeIndex() const {
    const auto& inputItr = this->inputsInfo.cbegin();
    if (inputItr == this->inputsInfo.cend()) {
        throw std::logic_error("model has no inputs");
    }
    const auto& input = inputItr->second;
    const auto batchIndex = input->getLayout().getBatchIndex();
    if (!batchIndex.has_value()) {
        throw std::logic_error("cannot get batch index");
    }
    return batchIndex.value();
}

uint32_t ModelInstance::getOptimalNumberOfInferRequests() const {
    try {
        OV_LOGGER("compiledModel: {}, ompiledModel->get_property(ov::optimal_number_of_infer_requests)", reinterpret_cast<const void*>(this->compiledModel.get()));
        uint32_t numOptimalInferRequests = compiledModel->get_property(ov::optimal_number_of_infer_requests);
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Number of OpenVINO streams: {}", numOptimalInferRequests);
        return numOptimalInferRequests;
    } catch (...) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Unable to get information about number of optimal infer requests; model: {}; version: {}; device: {}", getName(), getVersion(), config.getTargetDevice());
    }
    return 0;
}

uint32_t ModelInstance::getNumOfStreams() const {
    try {
        OV_LOGGER("compiledModel: {}, ompiledModel->get_property(ov::num_streams)", reinterpret_cast<const void*>(this->compiledModel.get()));
        uint32_t numStreams = compiledModel->get_property(ov::num_streams);
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Number of OpenVINO streams: {}", numStreams);
        return numStreams;
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Unable to get property ov::num_streams device: {}. Number of streams will be set to number of optimal infer requests.", config.getTargetDevice());
    }
    return getOptimalNumberOfInferRequests();
}

std::unique_ptr<RequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>> ModelInstance::createRequestProcessor(const tensorflow::serving::PredictRequest*, tensorflow::serving::PredictResponse*) {
    return std::make_unique<RequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>>();
}
std::unique_ptr<RequestProcessor<KFSRequest, KFSResponse>> ModelInstance::createRequestProcessor(const KFSRequest*, KFSResponse*) {
    return std::make_unique<RequestProcessor<KFSRequest, KFSResponse>>();
}
std::unique_ptr<RequestProcessor<InferenceRequest, InferenceResponse>> ModelInstance::createRequestProcessor(const InferenceRequest*, InferenceResponse*) {
    return std::make_unique<RequestProcessor<InferenceRequest, InferenceResponse>>();
}

template Status ModelInstance::infer<InferenceRequest, InferenceResponse>(InferenceRequest const*, InferenceResponse*, std::unique_ptr<ModelInstanceUnloadGuard>&);

template <typename RequestType, typename ResponseType>
RequestProcessor<RequestType, ResponseType>::RequestProcessor() = default;
template <typename RequestType, typename ResponseType>
RequestProcessor<RequestType, ResponseType>::~RequestProcessor() = default;
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::extractRequestParameters(const RequestType* request) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::prepare() { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::preInferenceProcessing(ov::InferRequest& inferRequest) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::postInferenceProcessing(ResponseType* response, ov::InferRequest& inferRequest) { return StatusCode::OK; }
template <typename RequestType, typename ResponseType>
Status RequestProcessor<RequestType, ResponseType>::release() { return StatusCode::OK; }

template class RequestProcessor<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>;
template class RequestProcessor<KFSRequest, KFSResponse>;
}  // namespace ovms
