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
#pragma once

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "../status.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "packettypes.hpp"

#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/packet.h"
#pragma GCC diagnostic pop

namespace ovms {

Status deserializeInputSidePacketsImpl(
            std::map<std::string, mediapipe::Packet>&   inputSidePackets,
    const   KFSRequest&                                 request);

// Imitation of stream.Read(...)
bool waitForNewRequest(
    KFSServerReaderWriter&  serverReaderWriter,
    KFSRequest&             newRequest);

// Supports only 1 output, each output is sent separately
// Maybe be called from different threads, requires synchronization.
Status sendPacketImpl(
    const   std::string&                endpointName,
    const   std::string&                endpointVersion,
    const   std::string&                packetName,
    const   mediapipe_packet_type_enum  packetType,
    const   ::mediapipe::Packet&        packet,
            KFSServerReaderWriter&      serverReaderWriter);

// TODO: Needs to support multiple inputs, we support multiple inputs at once
Status deserializePacketImpl(
    std::shared_ptr<const KFSRequest>               request,
    std::function<Status(
        const   mediapipe::Packet&,
        const   std::string&)>&&                    fn);
            // callback to fire when packet is created? TODO

}  // namespace ovms
