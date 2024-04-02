//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "mediapipe_stream_impl.hpp"

#include <sstream>
#include <string>

#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "../logging.hpp"

#include "../kfs_frontend/kfs_utils.hpp"
#include "../tfs_frontend/tfs_utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#include "opencv2/opencv.hpp"

#if (PYTHON_DISABLE == 0)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../python/python_backend.hpp"
#include "../python/pythonnoderesources.hpp"
#include "src/python/ovms_py_tensor.hpp"
namespace py = pybind11;
#endif

namespace ovms {

#define MP_RETURN_ON_FAIL(code, message, errorCode)              \
    {                                                            \
        auto absStatus = code;                                   \
        if (!absStatus.ok()) {                                   \
            const std::string absMessage = absStatus.ToString(); \
            SPDLOG_DEBUG("{} {}", message, absMessage);          \
            return Status(errorCode, std::move(absMessage));     \
        }                                                        \
    }

/////// util ////////

#define SET_DATA_FROM_MP_TENSOR(TENSOR, VIEW_TYPE)                                                     \
    switch ((TENSOR)->element_type()) {                                                                \
    case mediapipe::Tensor::ElementType::kFloat32:                                                     \
    case mediapipe::Tensor::ElementType::kFloat16:                                                     \
        data = reinterpret_cast<void*>(const_cast<float*>((TENSOR)->VIEW_TYPE().buffer<float>()));     \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kUInt8:                                                       \
        data = reinterpret_cast<void*>(const_cast<uint8_t*>((TENSOR)->VIEW_TYPE().buffer<uint8_t>())); \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kInt8:                                                        \
        data = reinterpret_cast<void*>(const_cast<int8_t*>((TENSOR)->VIEW_TYPE().buffer<int8_t>()));   \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kInt32:                                                       \
        data = reinterpret_cast<void*>(const_cast<int32_t*>((TENSOR)->VIEW_TYPE().buffer<int32_t>())); \
        break;                                                                                         \
    case mediapipe::Tensor::ElementType::kBool:                                                        \
        data = reinterpret_cast<void*>(const_cast<bool*>((TENSOR)->VIEW_TYPE().buffer<bool>()));       \
        break;                                                                                         \
    default:                                                                                           \
        data = reinterpret_cast<void*>(const_cast<void*>((TENSOR)->VIEW_TYPE().buffer<void>()));       \
    }

#define HANDLE_PACKET_RECEIVAL_EXCEPTIONS()                           \
    catch (const std::exception& e) {                                 \
        std::stringstream ss;                                         \
        ss << "Failed to get packet"                                  \
           << outputStreamName                                        \
           << " with exception: "                                     \
           << e.what();                                               \
        std::string details{ss.str()};                                \
        SPDLOG_DEBUG(details);                                        \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details)); \
    }                                                                 \
    catch (...) {                                                     \
        std::stringstream ss;                                         \
        ss << "Failed to get packet"                                  \
           << outputStreamName                                        \
           << " with exception.";                                     \
        std::string details{ss.str()};                                \
        SPDLOG_DEBUG(details);                                        \
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details)); \
    }

const KFSDataType EMPTY_PREC = "";

static const KFSDataType& MPPrecisionToKFSPrecision(::mediapipe::Tensor::ElementType precision) {
    static std::unordered_map<mediapipe::Tensor::ElementType, KFSDataType> precisionMap{
        //        {mediapipe::Tensor::ElementType::, "FP64"},
        {mediapipe::Tensor::ElementType::kFloat32, "FP32"},
        {mediapipe::Tensor::ElementType::kFloat16, "FP16"},
        //        {mediapipe::Tensor::ElementType::, "INT64"},
        {mediapipe::Tensor::ElementType::kInt32, "INT32"},
        //        {mediapipe::Tensor::ElementType::, "INT16"},
        {mediapipe::Tensor::ElementType::kInt8, "INT8"},
        //        {mediapipe::Tensor::ElementType::, "UINT64"},
        //        {mediapipe::Tensor::ElementType::, "UINT32"},
        //        {mediapipe::Tensor::ElementType::, "UINT16"},
        {mediapipe::Tensor::ElementType::kUInt8, "UINT8"},
        {mediapipe::Tensor::ElementType::kBool, "BOOL"}
        //        {"", ov::element::Type_t::, mediapipe::Tensor::ElementType::kChar}
    };
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        SPDLOG_WARN("Unsupported precision passed from Mediapipe graph");
        return EMPTY_PREC;
    }
    return it->second;
}

template <typename T>
static Status receiveAndSerializePacket(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName);

template <>
Status receiveAndSerializePacket<tensorflow::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<tensorflow::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                TFSPrecisionToOvmsPrecision(
                    received.dtype())));
        output->clear_shape();
        for (const auto& dim : received.shape()) {
            output->add_shape(dim.size);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.TotalBytes());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<::mediapipe::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const ::mediapipe::Tensor& received = packet.Get<::mediapipe::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(MPPrecisionToKFSPrecision(received.element_type()));
        output->clear_shape();
        for (const auto& dim : received.shape().dims) {
            output->add_shape(dim);
        }
        void* data;
        SET_DATA_FROM_MP_TENSOR(&received, GetCpuReadView);
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(data), received.bytes());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<ov::Tensor>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<ov::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                ovElementTypeToOvmsPrecision(
                    received.get_element_type())));
        output->clear_shape();
        for (const auto& dim : received.get_shape()) {
            output->add_shape(dim);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.get_byte_size());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

template <>
Status receiveAndSerializePacket<KFSResponse>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<KFSResponse>();
        response = std::move(received);
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

static KFSDataType convertImageFormatToKFSDataType(const mediapipe::ImageFormat::Format& imageFormat) {
    static std::unordered_map<mediapipe::ImageFormat::Format, KFSDataType> ImageFormatKFSDatatypeMap{
        {mediapipe::ImageFormat::GRAY8, "UINT8"},
        {mediapipe::ImageFormat::SRGB, "UINT8"},
        {mediapipe::ImageFormat::SRGBA, "UINT8"},
        {mediapipe::ImageFormat::GRAY16, "UINT16"},
        {mediapipe::ImageFormat::SRGB48, "UINT16"},
        {mediapipe::ImageFormat::SRGBA64, "UINT16"},
        {mediapipe::ImageFormat::VEC32F1, "FP32"},
        {mediapipe::ImageFormat::VEC32F2, "FP32"}};
    auto it = ImageFormatKFSDatatypeMap.find(imageFormat);
    if (it == ImageFormatKFSDatatypeMap.end()) {
        SPDLOG_DEBUG("Converting Mediapipe::ImageFrame format to KFS datatype failed. Datatype will be set to default - UINT8");
        return "UINT8";
    }
    return it->second;
}

template <>
Status receiveAndSerializePacket<mediapipe::ImageFrame>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const auto& received = packet.Get<mediapipe::ImageFrame>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        KFSDataType datatype = convertImageFormatToKFSDataType(received.Format());
        output->set_datatype(datatype);
        output->clear_shape();
        output->add_shape(received.Height());
        output->add_shape(received.Width());
        output->add_shape(received.NumberOfChannels());
        cv::Mat image = mediapipe::formats::MatView(&received);

        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(image.data), image.cols * image.rows * image.channels() * image.elemSize1());
        return StatusCode::OK;
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}

#if (PYTHON_DISABLE == 0)
template <>
Status receiveAndSerializePacket<PyObjectWrapper<py::object>>(const ::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        const PyObjectWrapper<py::object>& pyOutput = packet.Get<PyObjectWrapper<py::object>>();
        auto* output = response.add_outputs();
        output->set_name(pyOutput.getProperty<std::string>("name"));
        output->set_datatype(pyOutput.getProperty<std::string>("datatype"));
        output->clear_shape();
        for (const auto& dim : pyOutput.getProperty<std::vector<py::ssize_t>>("shape")) {
            output->add_shape(dim);
        }
        void* ptr = pyOutput.getProperty<void*>("ptr");
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(ptr), pyOutput.getProperty<py::ssize_t>("size"));
        return StatusCode::OK;
    } catch (const pybind11::error_already_set& e) {
        std::stringstream ss;
        ss << "Failed to get packet " << outputStreamName << " due to Python object unpacking error: " << e.what();
        std::string details{ss.str()};
        SPDLOG_DEBUG(details);
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
    }
    HANDLE_PACKET_RECEIVAL_EXCEPTIONS();
}
#endif

////////////////////

template <typename T, template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::Timestamp& timestamp, PythonBackend* pythonBackend) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    if (request->raw_input_contents().size() == 0) {
        const std::string details = "Invalid message structure - raw_input_content is empty";
        SPDLOG_DEBUG("[servable name: {} version: {}] {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    if (request->raw_input_contents().size() != request->inputs().size()) {
        std::stringstream ss;
        ss << "Size of raw_input_contents: " << request->raw_input_contents().size() << " is different than number of inputs: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request->model_name(), request->model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    std::unique_ptr<T> inputTensor;
    //OVMS_RETURN_ON_FAIL(deserializeTensor(name, *request, inputTensor, pythonBackend));
    MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                          name,
                          ::mediapipe::packet_internal::Create(
                              new Holder<T>(inputTensor.release(), request))
                              .At(timestamp)),
        std::string("failed to add packet to stream: ") + name, StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    return StatusCode::OK;
}

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& name, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::Timestamp& timestamp, PythonBackend* pythonBackend) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Request to passthrough:\"{}\"", name);
    const KFSRequest* lvaluePtr = request.get();
    MP_RETURN_ON_FAIL(graph.AddPacketToInputStream(
                          name,
                          ::mediapipe::packet_internal::Create(
                              new Holder<const KFSRequest*>(lvaluePtr, request))
                              .At(timestamp)),
        std::string("failed to add packet to stream: ") + name, StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
    return StatusCode::OK;
}

///// utils /////

// Two types of holders
// One (HolderWithRequestOwnership) is required for streaming where it is OVMS who creates the request but it is not the packet type and we have to clean up
// Second (HolderWithNoRequestOwnership) is required for unary-unary where it is gRPC who creates the request and musn't clean up
// Specializations are for special case when the request itsef is the packet and we need to ensure there is no double free
template <typename T>
class HolderWithRequestOwnership : public ::mediapipe::packet_internal::Holder<T> {
    std::shared_ptr<const KFSRequest> req;

public:
    explicit HolderWithRequestOwnership(const T* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::Holder<T>(barePtr),
        req(req) {}
};
template <>
class HolderWithRequestOwnership<const KFSRequest*> : public ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*> {
    const KFSRequest* hiddenPtr = nullptr;
    std::shared_ptr<const KFSRequest> req;

public:
    explicit HolderWithRequestOwnership(const KFSRequest* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*>(&hiddenPtr),
        hiddenPtr(barePtr),
        req(req) {}
};

template <typename T>
class HolderWithNoRequestOwnership : public ::mediapipe::packet_internal::Holder<T> {
public:
    explicit HolderWithNoRequestOwnership(const T* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::Holder<T>(barePtr) {}
};
template <>
class HolderWithNoRequestOwnership<const KFSRequest*> : public ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*> {
public:
    const KFSRequest* hiddenPtr = nullptr;
    explicit HolderWithNoRequestOwnership(const KFSRequest* barePtr, const std::shared_ptr<const KFSRequest>& req) :
        ::mediapipe::packet_internal::ForeignHolder<const KFSRequest*>(&hiddenPtr),
        hiddenPtr(barePtr) {}
};

template <template <typename X> typename Holder>
static Status createPacketAndPushIntoGraph(const std::string& inputName, std::shared_ptr<const KFSRequest>& request, ::mediapipe::CalculatorGraph& graph, const ::mediapipe::
Timestamp& timestamp, const stream_types_mapping_t& inputTypes, PythonBackend* pythonBackend) {
    auto inputPacketType = inputTypes.at(inputName);
    ovms::Status status;
    if (inputPacketType == mediapipe_packet_type_enum::KFS_REQUEST) {
        SPDLOG_DEBUG("Request processing KFS passthrough: {}", inputName);
        status = createPacketAndPushIntoGraph<Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::TFTENSOR) {
        SPDLOG_DEBUG("Request processing TF tensor: {}", inputName);
        status = createPacketAndPushIntoGraph<tensorflow::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::MPTENSOR) {
        SPDLOG_DEBUG("Request processing MP tensor: {}", inputName);
        status = createPacketAndPushIntoGraph<mediapipe::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    } else if (inputPacketType == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
        SPDLOG_DEBUG("Request processing Mediapipe ImageFrame: {}", inputName);
        status = createPacketAndPushIntoGraph<mediapipe::ImageFrame, Holder>(inputName, request, graph, timestamp, nullptr);
#if (PYTHON_DISABLE == 0)
    } else if (inputPacketType == mediapipe_packet_type_enum::OVMS_PY_TENSOR) {
        SPDLOG_DEBUG("Request processing OVMS Python input: {}", inputName);
        status = createPacketAndPushIntoGraph<PyObjectWrapper<py::object>, Holder>(inputName, request, graph, timestamp, pythonBackend);
#endif
    } else if ((inputPacketType == mediapipe_packet_type_enum::OVTENSOR) ||
               (inputPacketType == mediapipe_packet_type_enum::UNKNOWN)) {
        SPDLOG_DEBUG("Request processing OVTensor: {}", inputName);
        status = createPacketAndPushIntoGraph<ov::Tensor, Holder>(inputName, request, graph, timestamp, nullptr);
    }
    return status;
}

///////////////////


Status deserializeInputSidePacketsFromFirstRequestImpl(
            std::map<std::string, mediapipe::Packet>&   inputSidePackets,
    const   KFSRequest&                                 request) {

    static const std::string PYTHON_SESSION_SIDE_PACKET_TAG{"py"};  // TODO
    static const std::string TIMESTAMP_PARAMETER_NAME{"OVMS_MP_TIMESTAMP"};  // TODO
    for (const auto& [name, valueChoice] : request.parameters()) {
        SPDLOG_DEBUG("Found: {}; parameter in request for: {};", name, request.model_name());
        if (name == TIMESTAMP_PARAMETER_NAME) {
            SPDLOG_DEBUG("Ignored: {}; parameter in request for: {}; Paremeter is reserved for MediaPipe input packet timestamps", name, request.model_name());
            continue;
        }
        if (name == PYTHON_SESSION_SIDE_PACKET_TAG) {
            const std::string absMessage = "Incoming input side packet: " + PYTHON_SESSION_SIDE_PACKET_TAG + " is special reserved name and cannot be used";
            SPDLOG_DEBUG("Failed to insert predefined input side packet: {} with error: {}", PYTHON_SESSION_SIDE_PACKET_TAG, absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
        }
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param());
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param());
        } else {
            SPDLOG_DEBUG("Handling parameters of other types than: bool, string, int64 is not supported");
            return Status(StatusCode::NOT_IMPLEMENTED, "Handling parameters of other types than: bool, string, int64 is not supported");
        }
    }
    return StatusCode::OK;
}

bool waitForNewRequest(
    KFSServerReaderWriter& serverReaderWriter,
    KFSRequest& newRequest) {
    return serverReaderWriter.Read(&newRequest);
}

Status sendPacketImpl(
    const   std::string&                endpointName,
    const   std::string&                endpointVersion,
    const   std::string&                packetName,
    const   mediapipe_packet_type_enum  packetType,
    const   ::mediapipe::Packet&        packet,
            KFSServerReaderWriter&      serverReaderWriter) {

    KFSStreamResponse resp;

    Status status;
    SPDLOG_DEBUG("Received packet from output stream: {}", packetName);
    if (packetType == mediapipe_packet_type_enum::KFS_RESPONSE) {
        SPDLOG_DEBUG("Response processing packet type KFSPass name: {}", packetName);
        status = receiveAndSerializePacket<KFSResponse>(packet, *resp.mutable_infer_response(), packetName);
    } else if (packetType == mediapipe_packet_type_enum::TFTENSOR) {
        SPDLOG_DEBUG("Response processing packet type TF Tensor name: {}", packetName);
        status = receiveAndSerializePacket<tensorflow::Tensor>(packet, *resp.mutable_infer_response(), packetName);
    } else if (packetType == mediapipe_packet_type_enum::TFLITETENSOR) {
        SPDLOG_DEBUG("Response processing packet type TFLite Tensor name: {}", packetName);
        std::string details{"Response processing packet type TFLite Tensor is not supported"};
        status = Status(StatusCode::NOT_IMPLEMENTED, std::move(details));
    } else if (packetType == mediapipe_packet_type_enum::MPTENSOR) {
        SPDLOG_DEBUG("Response processing packet type MP Tensor name: {}", packetName);
        status = receiveAndSerializePacket<mediapipe::Tensor>(packet, *resp.mutable_infer_response(), packetName);
    } else if (packetType == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
        SPDLOG_DEBUG("Response processing Mediapipe Image Frame: {}", packetName);
        status = receiveAndSerializePacket<mediapipe::ImageFrame>(packet, *resp.mutable_infer_response(), packetName);
#if (PYTHON_DISABLE == 0)
    } else if (packetType == mediapipe_packet_type_enum::OVMS_PY_TENSOR) {
        SPDLOG_DEBUG("Response processing Ovms Python Tensor name: {}", packetName);
        status = receiveAndSerializePacket<PyObjectWrapper<py::object>>(packet, *resp.mutable_infer_response(), packetName);
#endif
    } else if ((packetType == mediapipe_packet_type_enum::OVTENSOR) ||
               (packetType == mediapipe_packet_type_enum::UNKNOWN)) {
        SPDLOG_DEBUG("Response processing packet type:  OVTensor name: {}", packetName);
        status = receiveAndSerializePacket<ov::Tensor>(packet, *resp.mutable_infer_response(), packetName);
    } else {
        status = Status(StatusCode::UNKNOWN_ERROR, "Unreachable code");
    }
    if (!status.ok()) {
        return status;
    }

    static const std::string TIMESTAMP_PARAMETER_NAME{"OVMS_MP_TIMESTAMP"};  // TODO
    *resp.mutable_infer_response()->mutable_model_name() = endpointName;
    *resp.mutable_infer_response()->mutable_model_version() = endpointVersion;
    resp.mutable_infer_response()->mutable_parameters()->operator[](TIMESTAMP_PARAMETER_NAME).set_int64_param(packet.Timestamp().Value());

    if (!serverReaderWriter.Write(resp)) {
        return Status(StatusCode::UNKNOWN_ERROR, "client disconnected");
    }

    return StatusCode::OK;
}

static Status deserializeTimestampIfAvailable(
    const KFSRequest& request,
    ::mediapipe::Timestamp& timestamp) {
    static const std::string TIMESTAMP_PARAMETER_NAME = "OVMS_MP_TIMESTAMP";
    auto timestampParamIt = request.parameters().find(TIMESTAMP_PARAMETER_NAME);
    if (timestampParamIt != request.parameters().end()) {
        SPDLOG_DEBUG("Found {} timestamp parameter in request for: {}", TIMESTAMP_PARAMETER_NAME, request.model_name());
        auto& parameterChoice = timestampParamIt->second;
        if (parameterChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            // Cannot create with error checking since error check = abseil death test
            timestamp = ::mediapipe::Timestamp::CreateNoErrorChecking(parameterChoice.int64_param());
            if (!timestamp.IsRangeValue()) {
                SPDLOG_DEBUG("Timestamp not in range: {}; for request to: {};", timestamp.DebugString(), request.model_name());
                return Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, timestamp.DebugString());
            }
        } else {
            auto status = Status(StatusCode::MEDIAPIPE_INVALID_TIMESTAMP, "Invalid timestamp format in request parameter OVMS_MP_TIMESTAMP. Should be int64");
            SPDLOG_DEBUG(status.string());
            return status;
        }
    }
    return StatusCode::OK;
}

Status recvPacketImpl(
    std::shared_ptr<const KFSRequest>   request,
    stream_types_mapping_t&             inputTypes,
    PythonBackend*                      pythonBackend,
    ::mediapipe::CalculatorGraph&       graph,
    ::mediapipe::Timestamp&             currentTimestamp) {

    auto status = deserializeTimestampIfAvailable(*request, currentTimestamp);
    if (!status.ok()) {
        return status;
    }

    for (const auto& input : request->inputs()) {
        const auto& inputName = input.name();
        auto status = createPacketAndPushIntoGraph<HolderWithRequestOwnership>(
            inputName, request, graph, currentTimestamp, inputTypes, pythonBackend);
        if (!status.ok()) {
            return status;
        }
    }

    currentTimestamp = currentTimestamp.NextAllowedInStream();

    return StatusCode::OK;
}

Status validateSubsequentRequestImpl(
    const KFSRequest& request,
    const std::string& endpointName,
    const std::string& endpointVersion,
    stream_types_mapping_t& inputTypes) {

    if (request.model_name() != endpointName) {
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_NAME;
    }
    if (request.model_version() != endpointVersion &&
        request.model_version() != "0" &&    // default version does not matter for user
        !request.model_version().empty()) {  // empty the same as default
        return StatusCode::MEDIAPIPE_INCORRECT_SERVABLE_VERSION;
    }

    return StatusCode::OK;
}

Status sendErrorImpl(
    KFSServerReaderWriter& serverReaderWriter,
    const std::string& message) {

    ::inference::ModelStreamInferResponse resp;
    *resp.mutable_error_message() = message;

    if (serverReaderWriter.Write(resp)) {
        return StatusCode::OK;
    }

    return Status(StatusCode::UNKNOWN_ERROR, "error during sending an error response");
}


}  // namespace ovms
