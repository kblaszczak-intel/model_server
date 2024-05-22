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
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

#include <openvino/openvino.hpp>
#include <continuous_batching_pipeline.hpp>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "llmnoderesources.hpp"

using namespace rapidjson;

namespace mediapipe {

// TODO: To be changed after mkulakow PR
using InputDataType = std::string;
using OutputDataType = std::string;

const std::string LLM_SESSION_SIDE_PACKET_TAG = "LLM_NODE_RESOURCES";

class HttpLLMCalculator : public CalculatorBase {
    std::shared_ptr<ovms::LLMNodeResources> nodeResources;
    std::shared_ptr<GenerationHandle> generationHandle;

    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    mediapipe::Timestamp timestamp{0};
    std::string body;
    std::chrono::time_point<std::chrono::system_clock> created;

    std::string serialize(const std::string& completion) {
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        writer.StartObject();
        
            writer.String("choices");
            writer.StartArray();
                writer.StartObject();

                    writer.String("finish_reason");
                    writer.String("stop");

                    writer.String("index");
                    writer.Int(0);

                    writer.String("message");
                    writer.StartObject();
                        writer.String("content");
                        writer.String(completion.c_str());  // TODO?
                        
                        writer.String("role");
                        writer.String("assistant");  // TODO?
                    writer.EndObject();
                    
                    writer.String("logprobs");
                    writer.Null();
                
                writer.EndObject();
            writer.EndArray();

            writer.String("created");
            writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

            // TODO: id
            // TODO: model
            
            writer.String("object");
            writer.String("chat.completion");

            // TODO: usge

        writer.EndObject();
        return buffer.GetString();
    }

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<InputDataType>();
        cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<OutputDataType>();
        cc->Outputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        return absl::OkStatus();
        // // Should probably also be moved to separate calculator
        // // Pros: separate logic
        // // Cons: Large pbtxt file

        // // Writing JSON
        // StringBuffer buffer;
        // Writer<StringBuffer> writer(buffer);

        // writer.StartObject();
        // writer.String("id");  // create Id property
        // writer.Int(3);        // write the Id value from the object instance
        // writer.EndObject();
        // ////////////

        // LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract start --" << buffer.GetString();
        // RET_CHECK(!cc->Inputs().GetTags().empty());
        // RET_CHECK(!cc->Outputs().GetTags().empty());

        // cc->Inputs().Tag("REQUEST").Set<std::string>();  // TODO: To be changed with HttpPayload? Or yet another data type (LLMData)?
        // cc->Outputs().Tag("RESPONSE").Set<std::string>();  // TODO: To be changed with yet another type and serialization moved to another calculator?

        // cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Set<ovms::LLMNodeResourcesMap>();
        // LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        // return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ovms::LLMNodeResourcesMap nodeResourcesMap = cc->InputSidePackets().Tag(LLM_SESSION_SIDE_PACKET_TAG).Get<ovms::LLMNodeResourcesMap>();
        auto it = nodeResourcesMap.find(cc->NodeName());
        RET_CHECK(it != nodeResourcesMap.end()) << "Could not find initialized LLM node named: " << cc->NodeName();

        nodeResources = it->second;
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

    // greedy (no params)
    // beam-search (num groups, num beams, penalties)
    // random sampling - in-progress (top-p top-k)
    absl::Status Process(CalculatorContext* cc) final {
        // For cases where MediaPipe decides to trigger Process() when there are no inputs
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }

        // First iteration of Process()
        if (!cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            // created time
            this->created = std::chrono::system_clock::now();

            std::string prompt = cc->Inputs().Tag(INPUT_TAG_NAME).Get<InputDataType>();
            LOG(INFO) << "Prompt: " << prompt;
            GenerationConfig config = GenerationConfig::greedy();
            GenerationHandle handle = nodeResources->cbPipe->add_request(0/*to be removed from API?*/, prompt, config);
            this->generationHandle = std::make_shared<GenerationHandle>(handle);
        }

        // unary only
        std::vector<GenerationOutput> generationOutput = this->generationHandle->read_all();
        RET_CHECK(generationOutput.size() == 1);

        std::vector<int64_t> tokens = generationOutput[0].generated_token_ids;
        std::shared_ptr<Tokenizer> tokenizer = nodeResources->cbPipe->get_tokenizer();
        std::string completion = tokenizer->decode(tokens);

        auto ff = serialize(completion);
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(new OutputDataType{ff}, timestamp);

        // Produce loopback in case of looping
        // if (std::find(this->body.begin(), this->body.end(), '8') == this->body.end()) {
        //     cc->Outputs().Tag(LOOPBACK_TAG_NAME).Add(new bool{true}, timestamp);
        // }

        timestamp = timestamp.NextAllowedInStream();

        return absl::OkStatus();
    }
};

// TODO: Names to be decided
const std::string HttpLLMCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string HttpLLMCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string HttpLLMCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(HttpLLMCalculator);
}  // namespace mediapipe
