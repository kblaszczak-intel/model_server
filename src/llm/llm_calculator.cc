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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <memory>

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>

#include "llmnoderesources.hpp"
#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;

constexpr size_t BATCH_SIZE = 1;

namespace mediapipe {

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

class LLMCalculator : public CalculatorBase {
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    // The calculator manages timestamp for outputs to work independently of inputs
    // this way we can support timestamp continuity for more than one request in streaming scenario.
    mediapipe::Timestamp outputTimestamp;
    bool hasLoopback{false};

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse>();

        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        if (it == nodeResourcesMap.end()) {
            LOG(INFO) << "Could not find initialized LLM node named: " << cc->NodeName();
            RET_CHECK(false);
        }

        nodeResources = it->second;
        outputTimestamp = mediapipe::Timestamp(mediapipe::Timestamp::Unset());
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
        try {
            const KFSRequest* request = cc->Inputs().Tag("REQUEST").Get<const KFSRequest*>();
            // Hardcoded single input for data
            auto data = request->raw_input_contents().Get(0);
            std::string prompt = std::string(data.begin(), data.end());
            LOG(INFO) << "Received prompt: " << prompt << std::endl;
            std::vector<std::string> prompts = {prompt};
            std::string resultStr;

            // GenerationConfig::greedy(), GenerationConfig::multinomial(), GenerationConfig::beam_search()
            std::vector<GenerationConfig> sampling_params = {GenerationConfig::greedy()};
            std::vector<GenerationResult> generation_results = nodeResources->cbPipe->generate(prompts, sampling_params);
            for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
                const GenerationResult& generation_result = generation_results[request_id];
                for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
                    resultStr += generation_result.m_generation_ids[output_id];
                }
            }

            LOG(INFO) << "Received response: " << resultStr << std::endl;
            //--------------------------------------------
            auto response = std::make_unique<KFSResponse>();
            auto* responseOutput = response->add_outputs();
            responseOutput->set_name("output");
            responseOutput->set_datatype("BYTES");
            responseOutput->clear_shape();
            responseOutput->add_shape(resultStr.size());
            response->add_raw_output_contents()->assign(reinterpret_cast<char*>(resultStr.data()), resultStr.size());

            cc->Outputs().Tag("RESPONSE").AddPacket(MakePacket<KFSResponse>(*response).At(cc->InputTimestamp()));
        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution");
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution");
        }
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe
