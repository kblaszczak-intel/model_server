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

Status deserializeInputSidePacketsImpl(
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

Status deserializePacketImpl(
    const   KFSRequest&         request,
            std::string&        name,
            mediapipe::Packet&  packet) {

    // TODO: Implement
    return StatusCode::OK;
}

}  // namespace ovms
