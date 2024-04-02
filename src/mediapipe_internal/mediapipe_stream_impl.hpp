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
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#if (PYTHON_DISABLE == 0)
#include "../python/python_backend.hpp"
#endif

namespace ovms {

Status deserializeInputSidePacketsFromFirstRequestImpl(
            std::map<std::string, mediapipe::Packet>&   inputSidePackets,
    const   KFSRequest&                                 request);

// Imitation of stream.Read(...)
bool waitForNewRequest(
    KFSServerReaderWriter&  serverReaderWriter,
    KFSRequest&             newRequest);

// Supports only 1 output, each output is sent separately
// Maybe be called from different threads, requires synchronization.
// For now the synchronization is automatic, in graph executor
Status sendPacketImpl(
    const   std::string&                endpointName,
    const   std::string&                endpointVersion,
    const   std::string&                packetName,
    const   mediapipe_packet_type_enum  packetType,
    const   ::mediapipe::Packet&        packet,
            KFSServerReaderWriter&      serverReaderWriter);

// Deserialization and pushing into the graph
Status recvPacketImpl(
    std::shared_ptr<const KFSRequest>               request,
    stream_types_mapping_t&                         inputTypes,
    PythonBackend*                                  pythonBackend,
    ::mediapipe::CalculatorGraph&                   graph);

}  // namespace ovms
