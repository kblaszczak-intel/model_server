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
#include "modelinstance_executor.hpp"

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
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
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

const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;

template <typename RequestType>
const Status validate(ModelInstance& instance, const RequestType* request) {
    OVMS_PROFILE_FUNCTION();
    return request_validation_utils::validate(
        *request,
        instance.getInputsInfo(),
        instance.getName(),
        instance.getVersion(),
        instance.getOptionalInputNames(),
        instance.getModelConfig().getBatchingMode(),
        instance.getModelConfig().getShapes());
}

template const Status validate(ModelInstance&, const InferenceRequest*);
template const Status validate(ModelInstance&, const ::KFSRequest*);
template const Status validate(ModelInstance&, const tensorflow::serving::PredictRequest*);

template <typename RequestType, typename ResponseType>
Status infer(ModelInstance& instance, const RequestType* requestProto,
    ResponseType* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    using std::chrono::microseconds;

    auto requestProcessor = instance.createRequestProcessor(requestProto, responseProto);  // request, response passed only to deduce type
    auto status = requestProcessor->extractRequestParameters(requestProto);
    if (!status.ok())
        return status;
    status = validate(instance, requestProto);
    if (status.batchSizeChangeRequired() || status.reshapeRequired()) {
        // We are ensured that request shape is valid and convertible to model shape (non negative, non zero)
        // We can use it to perform reshape via shape=auto
        auto requestBatchSize = getRequestBatchSize(requestProto, instance.getBatchSizeIndex());
        auto requestShapes = getRequestShapes(requestProto);
        status = instance.reloadModelIfRequired(status, requestBatchSize, requestShapes, modelUnloadGuardPtr);
    }
    if (!status.ok())
        return status;
    status = requestProcessor->prepare();
    if (!status.ok())
        return status;

    timer.start(GET_INFER_REQUEST);
    OVMS_PROFILE_SYNC_BEGIN("getInferRequest");
    ExecutingStreamIdGuard executingStreamIdGuard(instance.getInferRequestsQueue(), instance.getMetricReporter());
    int executingInferId = executingStreamIdGuard.getId();
    ov::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    OVMS_PROFILE_SYNC_END("getInferRequest");
    timer.stop(GET_INFER_REQUEST);
    double getInferRequestTime = timer.elapsed<microseconds>(GET_INFER_REQUEST);
    OBSERVE_IF_ENABLED(instance.getMetricReporter().waitForInferReqTime, getInferRequestTime);
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, getInferRequestTime / 1000);

    timer.start(PREPROCESS);
    status = requestProcessor->preInferenceProcessing(inferRequest);
    timer.stop(PREPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Preprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(PREPROCESS) / 1000);

    timer.start(DESERIALIZE);
    InputSink<ov::InferRequest&> inputSink(inferRequest);
    bool isPipeline = false;
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, instance.getInputsInfo(), inputSink, isPipeline);
    timer.stop(DESERIALIZE);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(DESERIALIZE) / 1000);

    timer.start(PREDICTION);
    status = instance.performInference(inferRequest);
    timer.stop(PREDICTION);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(PREDICTION) / 1000);

    timer.start(SERIALIZE);
    OutputGetter<ov::InferRequest&> outputGetter(inferRequest);
    status = serializePredictResponse(outputGetter, instance.getName(), instance.getVersion(), instance.getOutputsInfo(), responseProto, getTensorInfoName, useSharedOutputContentFn(requestProto));
    timer.stop(SERIALIZE);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(SERIALIZE) / 1000);

    timer.start(POSTPROCESS);
    status = requestProcessor->postInferenceProcessing(responseProto, inferRequest);
    timer.stop(POSTPROCESS);
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Postprocessing duration in model {}, version {}, nireq {}: {:.3f} ms",
        instance.getName(), instance.getVersion(), executingInferId, timer.elapsed<microseconds>(POSTPROCESS) / 1000);
    if (instance.targetDevice == "AUTO")
        for (std::string device : instance.compiledModel->get_property(ov::execution_devices))
            SPDLOG_DEBUG("Used device: {}", device);

    status = requestProcessor->release();
    return status;
}
template Status infer<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>(ModelInstance&, const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);
template Status infer(ModelInstance&, const ::KFSRequest* requestProto,
    ::KFSResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

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
