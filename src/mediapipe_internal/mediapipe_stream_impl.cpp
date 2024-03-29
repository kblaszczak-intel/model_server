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

#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop

#include "../logging.hpp"

namespace ovms {

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

// Status sendPacketImpl(
//     const   std::string&            endpointName,
//     const   std::string&            endpointVersion,
//     const   std::string&            packetName,
//     const   ::mediapipe::Packet&    packet,
//             KFSServerReaderWriter&  serverReaderWriter) {

//     KFSStreamResponse resp;

//     // TODO: serializePacket

//     static const std::string TIMESTAMP_PARAMETER_NAME{"OVMS_MP_TIMESTAMP"};  // TODO
//     *resp.mutable_infer_response()->mutable_model_name() = endpointName;
//     *resp.mutable_infer_response()->mutable_model_version() = endpointVersion;
//     resp.mutable_infer_response()->mutable_parameters()->operator[](TIMESTAMP_PARAMETER_NAME).set_int64_param(packet.Timestamp().Value());

//     if (!serverReaderWriter.Write(resp)) {
//         return Status(StatusCode::UNKNOWN_ERROR, "client disconnected");
//     }

//     return StatusCode::OK;
// }

Status deserializePacketImpl(
    const   KFSRequest&         request,
            std::string&        name,
            mediapipe::Packet&  packet) {

    // TODO: Implement
    return StatusCode::OK;
}

}  // namespace ovms
