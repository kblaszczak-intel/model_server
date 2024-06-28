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

#include <algorithm>
#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <continuous_batching_pipeline.hpp>

#include "../logging.hpp"
#include "../profiler.hpp"

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#pragma GCC diagnostic pop

#include "../profiler.hpp"
#include "http_payload.hpp"

// Python execution for template processing
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "src/python/utils.hpp"

using namespace rapidjson;

namespace ovms {

// ----------------- TO BE MOVED TO SEPARATE FILES

static const std::string CHAT_TEMPLATE_WARNING_MESSAGE = "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint.";

struct TextProcessor {
    std::string bosToken = "";
    std::string eosToken = "";
    std::unique_ptr<PyObjectWrapper<py::object>> chatTemplate = nullptr;
};


enum class Endpoint {
    CHAT_COMPLETIONS,
    COMPLETIONS,
};

using chat_entry_t = std::unordered_map<std::string, std::string>;
using chat_t = std::vector<chat_entry_t>;

#define IGNORE_EOS_MAX_TOKENS_LIMIT 4000
class OpenAIChatCompletionsRequest {
    Document& doc;

    chat_t messages;
    std::optional<std::string> prompt{std::nullopt};
    bool stream{false};
    std::string model;
    std::optional<int> maxTokens{std::nullopt};
    // float frequencyPenalty{0.0f};
    // float presencePenalty{0.0f};
    std::optional<float> diversityPenalty{std::nullopt};
    std::optional<float> repetitionPenalty{std::nullopt};
    std::optional<float> lengthPenalty{std::nullopt};
    std::optional<int> numReturnSequences{std::nullopt};
    std::optional<float> temperature{std::nullopt};
    std::optional<float> topP{std::nullopt};
    std::optional<int> topK{std::nullopt};
    std::optional<int> seed{std::nullopt};
    std::optional<int> bestOf{std::nullopt};
    // std::optional<bool> useBeamSearch{std::nullopt};
    std::optional<bool> ignoreEOS{std::nullopt};
    Endpoint endpoint;

public:
    OpenAIChatCompletionsRequest(Document& doc, Endpoint endpoint) :
        doc(doc),
        endpoint(endpoint) {}

    GenerationConfig createGenerationConfig() const {
        GenerationConfig config;

        // Generic
        if (maxTokens.has_value())
            config.max_new_tokens = maxTokens.value();
        // TODO: max_length = ?
        if (ignoreEOS.has_value())
            config.ignore_eos = ignoreEOS.value();

        // Beam search specific
        config.num_groups = 1;  // OpenAI hardcoded
        if (bestOf.has_value())
            config.group_size = bestOf.value();
        if (diversityPenalty.has_value())
            config.diversity_penalty = diversityPenalty.value();  // TODO: Not available in OpenAI nor vLLM
        // TODO: stop_criteria = ?
        if (numReturnSequences.has_value())
            config.num_return_sequences = numReturnSequences.value();
        if (repetitionPenalty.has_value())
            config.repetition_penalty = repetitionPenalty.value();
        if (lengthPenalty.has_value())
            config.length_penalty = lengthPenalty.value();
        // TODO: no_repeat_ngram_size = ?
        // TODO: early_finish = ?
        // TODO use_beam_search is unused ?

        // Multinomial specific
        if (temperature.has_value())
            config.temperature = temperature.value();
        if (topK.has_value())
            config.top_k = topK.value();
        if (topP.has_value())
            config.top_p = topP.value();
        if (seed.has_value())
            config.rng_seed = seed.value();
        config.do_sample = config.temperature > 0.0f && config.group_size == 1;

        return config;
    }

    chat_t getMessages() const { return this->messages; }
    Endpoint getEndpoint() const { return this->endpoint; }
    std::optional<std::string> getPrompt() const { return this->prompt; }

    bool isStream() const { return this->stream; }
    std::string getModel() const { return this->model; }

    Status parse() {
        OVMS_PROFILE_FUNCTION();
        // stream: bool; optional
        if (!this->doc.IsObject())
            return StatusCode::OK;
            //return absl::InvalidArgumentError("Received json is not an object");
        auto it = this->doc.FindMember("stream");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsBool())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("Stream is not bool");
            this->stream = it->value.GetBool();
        }

        // messages: [{role: content}, {role: content}, ...]; required
        if (this->endpoint == Endpoint::CHAT_COMPLETIONS) {
            it = doc.FindMember("messages");
            if (it == doc.MemberEnd())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("Messages missing in request");
            if (!it->value.IsArray())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("Messages are not an array");
            this->messages.clear();
            this->messages.reserve(it->value.GetArray().Size());
            for (int i = 0; i < it->value.GetArray().Size(); i++) {
                const auto& obj = it->value.GetArray()[i];
                if (!obj.IsObject())
                    return StatusCode::OK;
                    //return absl::InvalidArgumentError("Message is not a JSON object");
                auto& chat = this->messages.emplace_back(chat_entry_t{});
                for (auto member = obj.MemberBegin(); member != obj.MemberEnd(); member++) {
                    if (!member->name.IsString())
                        return StatusCode::OK;
                        //return absl::InvalidArgumentError("Invalid message structure");
                    if (!member->value.IsString())
                        return StatusCode::OK;
                        //return absl::InvalidArgumentError("Invalid message structure");
                    chat[member->name.GetString()] = member->value.GetString();
                }
            }
        }

        // prompt: string
        if (this->endpoint == Endpoint::COMPLETIONS) {
            it = this->doc.FindMember("prompt");
            if (it != this->doc.MemberEnd()) {
                if (!it->value.IsString()) {
                    return StatusCode::OK;
                    //return absl::InvalidArgumentError("prompt is not a string");
                } else {
                    this->prompt = it->value.GetString();
                }
            }
        }
        // model: string; required
        it = this->doc.FindMember("model");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsString())
                return StatusCode::OK;    
                //return absl::InvalidArgumentError("model is not a string");
            this->model = it->value.GetString();
        } else {
            return StatusCode::OK;
            //return absl::InvalidArgumentError("model missing in request");
        }

        // ignore_eos: bool; optional - defaults to false
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("ignore_eos");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsBool())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("ignore_eos accepts values true or false");
            this->ignoreEOS = it->value.GetBool();
        }

        // max_tokens: int; optional
        it = this->doc.FindMember("max_tokens");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("max_tokens is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return StatusCode::OK;
                //return absl::InvalidArgumentError("max_tokens value should be greater than 0");
            this->maxTokens = it->value.GetUint();
        }
        if (this->ignoreEOS.value_or(false)) {
            if (this->maxTokens.has_value()) {
                if (it->value.GetUint() > IGNORE_EOS_MAX_TOKENS_LIMIT)
                    return StatusCode::OK;
                    //return absl::InvalidArgumentError("when ignore_eos is true max_tokens can not be greater than 4000");
            } else {
                this->maxTokens = IGNORE_EOS_MAX_TOKENS_LIMIT;
            }
        }
        // TODO: Supported by OpenAI and vLLM, however unsupported by CB lib
        // // frequency_penalty: float; optional - defaults to 0
        // it = this->doc.FindMember("frequency_penalty");
        // if (it != this->doc.MemberEnd()) {
        //     return false;  // TODO: Unsupported by CB
        //     if (!it->value.IsDouble())
        //         return false;
        //     this->frequencyPenalty = it->value.GetDouble();
        //     if (this->frequencyPenalty < -2.0f || this->frequencyPenalty > 2.0f)
        //         return false;
        // }

        // TODO: Supported by OpenAI and vLLM, however unsupported by CB lib
        // // presence_penalty: float; optional - defaults to 0
        // it = this->doc.FindMember("presence_penalty");
        // if (it != this->doc.MemberEnd()) {
        //     return false;  // TODO: Unsupported by CB
        //     if (!it->value.IsDouble())
        //         return false;
        //     this->presencePenalty = it->value.GetDouble();
        //     if (this->presencePenalty < -2.0f || this->presencePenalty > 2.0f)
        //         return false;
        // }

        // repetition_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("repetition_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("repetition_penalty is not a floating point number");
            this->repetitionPenalty = it->value.GetDouble();
        }

        // diversity_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API and vLLM, however available in CB lib
        it = this->doc.FindMember("diversity_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("diversity_penalty is not a floating point number");
            this->diversityPenalty = it->value.GetDouble();
        }

        // length_penalty: float; optional - defaults to 1.0
        // Extension, unsupported by OpenAI API however supported by vLLM and CB lib
        it = this->doc.FindMember("length_penalty");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("length_penalty is not a floating point number");
            this->lengthPenalty = it->value.GetDouble();
        }

        // temperature: float; optional - defaults to 0.0 (different than OpenAI which is 1.0)
        it = this->doc.FindMember("temperature");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("temperature is not a floating point number");
            this->temperature = it->value.GetDouble();
            if (this->temperature < 0.0f || this->temperature > 2.0f)
                return StatusCode::OK;
                //return absl::InvalidArgumentError("temperature out of range(0.0, 2.0)");
        }

        // top_p: float; optional - defaults to 1
        it = this->doc.FindMember("top_p");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsDouble())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("top_p is not a floating point number");
            this->topP = it->value.GetDouble();
            if (this->topP < 0.0f || this->topP > 1.0f)
                return StatusCode::OK;
                //return absl::InvalidArgumentError("top_p out of range(0.0, 1.0)");
        }

        // top_k: int; optional - defaults to 0
        // Extension, unsupported by OpenAI API, however supported by vLLM and CB lib
        it = this->doc.FindMember("top_k");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsInt())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("top_k is not an integer");
            this->topK = it->value.GetInt();
        }

        // seed: int; optional - defaults to 0 (not set)
        it = this->doc.FindMember("seed");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("seed is not an unsigned integer");
            this->seed = it->value.GetUint();
        }

        // best_of: int; optional - defaults to 1
        // Extension, unsupported by OpenAI API, however supported by vLLM, supported in CB lib by mapping to group_size param
        it = this->doc.FindMember("best_of");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("best_of is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return StatusCode::OK;
                //return absl::InvalidArgumentError("best_of value should be greater than 0");
            this->bestOf = it->value.GetUint();
        }

        // n: int; optional - defaults to 1
        // Supported by OpenAI API and vLLM, supported in CB lib by mapping to num_return_sequences param
        it = this->doc.FindMember("n");
        if (it != this->doc.MemberEnd()) {
            if (!it->value.IsUint())
                return StatusCode::OK;
                //return absl::InvalidArgumentError("n is not an unsigned integer");
            if (it->value.GetUint() == 0)
                return StatusCode::OK;
                //return absl::InvalidArgumentError("n value should be greater than 0");
            size_t bestOf = this->bestOf.has_value() ? this->bestOf.value() : 1;  // 1 is default best_of value
            if (bestOf < it->value.GetUint()) {
                return StatusCode::OK;
                //return absl::InvalidArgumentError("n value cannot be greater than best_of");
            }
            this->numReturnSequences = it->value.GetUint();
        }

        // use_beam_search: bool; optional - defaults to false
        // Extension from vLLM, unsupported by OpenAI API, not available directly in CB lib
        // Use best_of>1 to steer into beams earch
        // it = this->doc.FindMember("use_beam_search");
        // if (it != this->doc.MemberEnd()) {
        //     if (!it->value.IsBool())
        //         return false;
        //     this->useBeamSearch = it->value.GetBool();
        // }

        // logit_bias TODO
        // logprops TODO
        // top_logprobs TODO
        // response_format TODO
        // stop TODO
        // stream_options TODO
        // tools TODO
        // tool_choice TODO
        // user TODO
        // function_call TODO (deprecated)
        // functions TODO (deprecated)
        return StatusCode::OK;
    }
};

// TODO: To be moved to CB library.
class TextStreamer {
    std::shared_ptr<Tokenizer> tokenizer;
    std::vector<int64_t> tokenCache;
    size_t printLen{0};

public:
    TextStreamer(std::shared_ptr<Tokenizer> tokenizer) :
        tokenizer(tokenizer) {}

    std::optional<std::string> put(int64_t token) {
        tokenCache.push_back(token);
        std::string text = tokenizer->decode(tokenCache);

        if (!text.empty() && '\n' == text.back()) {
            // The chunk is ready if the generated text ends with new line.
            // Also, clear the cache.
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            tokenCache.clear();
            printLen = 0;
            return chunk;
        } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {  // NOLINT
            return std::nullopt;
        } else if (text.size() > printLen) {
            // The chunk is ready if the new text in the cache contains space.
            // The chunk is constructed from the new text, however only up to the last space character (including it)
            // Does not clear the cache.
            auto lastSpacePos = text.rfind(' ');
            if (lastSpacePos == std::string::npos || lastSpacePos < printLen) {
                return std::nullopt;
            }
            std::string chunk = std::string{text.data() + printLen, lastSpacePos - printLen + 1};
            printLen = lastSpacePos + 1;
            return chunk;
        }
        return std::nullopt;
    }
};

static std::string packIntoServerSideEventMessage(const std::string& message) {
    std::stringstream ss;
    ss << "data: " << message << "\n\n";
    return ss.str();
}

static std::string serializeUnaryResponse(const std::vector<std::string>& completeResponses, Endpoint endpoint, 
                                   OpenAIChatCompletionsRequest& request, std::chrono::time_point<std::chrono::system_clock>& created) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    writer.String("choices");
    writer.StartArray();  // [
    int i = 0;
    for (const std::string& completeResponse : completeResponses) {
        writer.StartObject();  // {
        // finish_reason: string; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)
        // "stop" => natural stop point due to stopping criteria <---------------- the only used so far, remaining are TODO
        // "length" => due to reaching max_tokens parameter TODO
        // "content_filter" => when produced restricted output
        // "tool_calls" => generation stopped and waiting for tool output
        // "function_call" => deprecated
        writer.String("finish_reason");
        writer.String("stop");
        // index: integer; Choice index, only n=1 supported anyway
        writer.String("index");
        writer.Int(i++);
        // logprobs: object/null; Log probability information for the choice. TODO
        writer.String("logprobs");
        writer.Null();
        // message: object
        if (endpoint == Endpoint::CHAT_COMPLETIONS) {
            writer.String("message");
            writer.StartObject();  // {
            // content: string; Actual content of the text produced
            writer.String("content");
            writer.String(completeResponse.c_str());
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String("role");
            writer.String("assistant");  // TODO - hardcoded
            // TODO: tools_call
            // TODO: function_call (deprecated)
            writer.EndObject();  // }
        } else if (endpoint == Endpoint::COMPLETIONS) {
            writer.String("text");
            writer.String(completeResponse.c_str());
        }

        writer.EndObject();  // }
    }
    writer.EndArray();  // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.getModel().c_str());

    // object: string; defined that the type is unary rather than streamed chunk
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion");
    }

    // TODO
    // id: string; A unique identifier for the chat completion.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // TODO
    // usage: object; Usage statistics for the completion request.
    // Might be crucial - possibly required for benchmarking purposes?

    writer.EndObject();  // }
    return buffer.GetString();
}

static std::string serializeUnaryResponse(const std::string& completeResponse, Endpoint endpoint, 
                                   OpenAIChatCompletionsRequest& request, std::chrono::time_point<std::chrono::system_clock>& created) {
    return serializeUnaryResponse(std::vector<std::string>{completeResponse}, endpoint, request, created);
}

static std::string serializeStreamingChunk(const std::string& chunkResponse, bool stop, Endpoint endpoint, OpenAIChatCompletionsRequest& request, std::chrono::time_point<std::chrono::system_clock>& created) {
    OVMS_PROFILE_FUNCTION();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);

    writer.StartObject();  // {

    // choices: array of size N, where N is related to n request parameter
    // Can also be empty for the last chunk if you set stream_options: {"include_usage": true} TODO
    writer.String("choices");
    writer.StartArray();   // [
    writer.StartObject();  // {
    // finish_reason: string or null; "stop"/"length"/"content_filter"/"tool_calls"/"function_call"(deprecated)/null
    // "stop" => natural stop point due to stopping criteria <---------------- the only used so far, remaining are TODO
    // "length" => due to reaching max_tokens parameter TODO
    // "content_filter" => when produced restricted output
    // "tool_calls" => generation stopped and waiting for tool output
    // "function_call" => deprecated
    // null - natural scenario when the generation has not completed yet
    writer.String("finish_reason");
    if (stop)
        writer.String("stop");
    else
        writer.Null();
    // index: integer; Choice index, only n=1 supported anyway
    writer.String("index");
    writer.Int(0);
    // logprobs: object/null; Log probability information for the choice. TODO
    writer.String("logprobs");
    writer.Null();
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("delta");
        writer.StartObject();  // {
        if (!stop) {
            writer.String("content");
            // writer.String("role");
            // writer.String("assistant");
            // role: string; Role of the text producer
            // Will make sense once we have chat templates? TODO(atobisze)
            writer.String(chunkResponse.c_str());
        }
        writer.EndObject();  // }
    } else if (endpoint == Endpoint::COMPLETIONS) {
        if (!stop) {
            writer.String("text");
            writer.String(chunkResponse.c_str());
        }
    }
    // TODO: tools_call
    // TODO: function_call (deprecated)
    writer.EndObject();  // }
    writer.EndArray();   // ]

    // created: integer; Unix timestamp (in seconds) when the MP graph was created.
    writer.String("created");
    writer.Int(std::chrono::duration_cast<std::chrono::seconds>(created.time_since_epoch()).count());

    // model: string; copied from the request
    writer.String("model");
    writer.String(request.getModel().c_str());

    // object: string; defined that the type streamed chunk rather than complete response
    if (endpoint == Endpoint::CHAT_COMPLETIONS) {
        writer.String("object");
        writer.String("chat.completion.chunk");
    } else if (endpoint == Endpoint::COMPLETIONS) {
        writer.String("object");
        writer.String("text_completion.chunk");
    }

    // TODO
    // id: string; A unique identifier for the chat completion. Each chunk has the same ID.

    // TODO
    // system_fingerprint: string; This fingerprint represents the backend configuration that the model runs with.
    // Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.

    // TODO
    // usage: object; An optional field that will only be present when you set stream_options: {"include_usage": true} in your request.
    // When present, it contains a null value except for the last chunk which contains the token usage statistics for the entire request.
    // Might be crucial - possibly required for benchmarking purposes?

    writer.EndObject();  // }
    return buffer.GetString();
}

// CB lib internals rely on request_id, so for now we provide increasing ID
static std::atomic<uint64_t> currentRequestId = 0;


// ------------------------------------------------

struct LLMExecutor {
    // For logging purposes we could have more information about graph and node here
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<ContinuousBatchingPipeline> pipe = nullptr;

    LLMExecutor(std::shared_ptr<ContinuousBatchingPipeline> pipe) {
        this->pipe = pipe;
    }

    bool hasRequests() {
        return (pipe->has_non_finished_requests());
    }

    void step() {
        OVMS_PROFILE_FUNCTION();
        pipe->step();
    }

    void waitForRequests(std::atomic<bool>* receivedEndSignal) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this, receivedEndSignal] { return (pipe->has_non_finished_requests() || *receivedEndSignal); });
    }

    void notify() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_one();
    }

    void printMetrics() {
        PipelineMetrics metrics = pipe->get_metrics();
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "All requests: {}; Scheduled requests: {}; Cache usage {:.1f}%;",
            metrics.requests, metrics.scheduled_requests, metrics.cache_usage * 100);
    }
};

class LLMExecutorWrapper {
    LLMExecutor llmExecutor;
    std::thread llmExecutorThread;
    std::atomic<bool> finishExecutorThread = false;
    TextProcessor textProcessor;

    static void run(LLMExecutor* llmExecutor, std::atomic<bool>* receivedEndSignal) {
        while (!(*receivedEndSignal)) {
            try {
                llmExecutor->printMetrics();
                if (llmExecutor->hasRequests()) {
                    llmExecutor->step();
                } else {
                    llmExecutor->waitForRequests(receivedEndSignal);
                }
            } catch (std::exception& e) {
                SPDLOG_LOGGER_ERROR(llm_executor_logger, "Error occurred in LLM executor: {}.", e.what());
                exit(1);
            }
        }
    }

public:
    LLMExecutorWrapper(std::shared_ptr<ContinuousBatchingPipeline> pipe) :
        llmExecutor(pipe) {
        loadTextProcessor();
        llmExecutorThread = std::thread(LLMExecutorWrapper::run, &llmExecutor, &finishExecutorThread);
    }

    ~LLMExecutorWrapper() {
        finishExecutorThread = true;
        llmExecutor.notify();
        llmExecutorThread.join();
    }

    void notifyNewRequestArrived() {
        llmExecutor.notify();
    }

    void loadTextProcessor() {
        py::gil_scoped_acquire acquire;
        try {
            auto locals = py::dict("models_path"_a = "/ovms/Meta-Llama-3-8B-Instruct");
            py::exec(R"(
                # Following the logic from:
                # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/tokenization_utils_base.py#L1837

                global json
                import json
                from pathlib import Path

                global jinja2
                import jinja2
                from jinja2.sandbox import ImmutableSandboxedEnvironment

                def raise_exception(message):
                    raise jinja2.exceptions.TemplateError(message)


                # Default chat template accepts only single message and outputs only it's 'content'
                # effectively turning it into a regular prompt. 
                default_chat_template = "{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"

                bos_token = ""
                eos_token = ""
                chat_template = default_chat_template

                template = None

                # Try to read template from template.jinja file
                jinja_file = Path(models_path + "/template.jinja")
                if jinja_file.is_file():
                    template_loader = jinja2.FileSystemLoader(searchpath=models_path)
                    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, loader=template_loader)
                    jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                    jinja_env.globals["raise_exception"] = raise_exception
                    template = jinja_env.get_template("template.jinja")

                # Try to read data from tokenizer_config.json
                tokenizer_config_file = Path(models_path + "/tokenizer_config.json")
                if tokenizer_config_file.is_file():
                    f = open(models_path + "/tokenizer_config.json")
                    data = json.load(f)
                    bos_token = data.get("bos_token", "")
                    eos_token = data.get("eos_token", "")
                    chat_template = data.get("chat_template", default_chat_template)

                if template is None:
                    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                    jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
                    jinja_env.globals["raise_exception"] = raise_exception
                    template = jinja_env.from_string(chat_template)
            )",
                py::globals(), locals);

            textProcessor.bosToken = locals["bos_token"].cast<std::string>();
            textProcessor.eosToken = locals["eos_token"].cast<std::string>();
            textProcessor.chatTemplate = std::make_unique<PyObjectWrapper<py::object>>(locals["template"]);
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
            SPDLOG_DEBUG("Chat template loading failed with error: {}", e.what());
        } catch (...) {
            SPDLOG_INFO(CHAT_TEMPLATE_WARNING_MESSAGE);
            SPDLOG_DEBUG("Chat template loading failed with an unexpected error");
        }
    }

    bool applyChatTemplate(std::string& requestBody, std::string& output) {
        if (textProcessor.chatTemplate == nullptr) {
            output = "Error: Chat template not loaded correctly, so it cannot be applied";
            return false;
        }

        py::gil_scoped_acquire acquire;
        try {
            auto locals = py::dict("request_body"_a = requestBody, "chat_template"_a = textProcessor.chatTemplate->getObject(),
                "bos_token"_a = textProcessor.bosToken, "eos_token"_a = textProcessor.eosToken);
            py::exec(R"(
                output = ""
                error = ""
                try:
                    messages = json.loads(request_body)["messages"]
                    output = chat_template.render(messages=messages, bos_token=bos_token, eos_token=eos_token, add_generation_prompt=True)
                except Exception as e:
                    error = str(e)            
            )",
                py::globals(), locals);

            std::string result = locals["output"].cast<std::string>();
            std::string error = locals["error"].cast<std::string>();

            if (error != "") {
                output = error;
                return false;
            }

            output = result;
            return true;
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_DEBUG("Error occured when applying chat template: {}", e.what());
            output = "Unexpected error occurred when applying chat template";
        } catch (...) {
            SPDLOG_DEBUG("Unexpected error occurred when applying chat template");
            output = "Unexpected error occurred when applying chat template";
        }
        return false;
    }

    Status infer(HttpPayload& request, std::string& response) {
        // Register resource creation time
        auto created = std::chrono::system_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Request body: {}", request.body);
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Request uri: {}", request.uri);
        Endpoint endpoint;
        if (request.uri == "/v3/chat/completions") {
            endpoint = Endpoint::CHAT_COMPLETIONS;
        } else if (request.uri == "/v3/completions") {
            endpoint = Endpoint::COMPLETIONS;
        } else {
            //return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
            return StatusCode::INVALID_MESSAGE_STRUCTURE;
        }
        auto openAIRequest = std::make_shared<OpenAIChatCompletionsRequest>(*request.parsedJson, endpoint);

        auto status = openAIRequest->parse();
        if (!status.ok())
            return status;

        std::string finalPrompt = "";

        // LOG(INFO) << "Input prompt:" << templateApplyOutput;

        std::string prompt;
        switch (endpoint) {
        case Endpoint::CHAT_COMPLETIONS: {
            if (openAIRequest->getMessages().size() <= 0) {
                //return absl::Status(absl::StatusCode::kInvalidArgument, "There are no messages to apply for chat");
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            if (!applyChatTemplate(request.body, finalPrompt)) {
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            break;
        }
        case Endpoint::COMPLETIONS: {
            if (!openAIRequest->getPrompt().has_value() || !openAIRequest->getPrompt().value().size()) {
                //return absl::Status(absl::StatusCode::kInvalidArgument, "Prompt is missing");
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            finalPrompt = openAIRequest->getPrompt().value();
        }
        }

        auto generationHandle = llmExecutor.pipe->add_request(
            currentRequestId++, /*to be removed from API?*/
            finalPrompt,
            openAIRequest->createGenerationConfig());

        notifyNewRequestArrived();

        std::vector<GenerationOutput> generationOutput = generationHandle->read_all();

        assert(generationOutput.size() >= 1);
        auto tokenizer = llmExecutor.pipe->get_tokenizer();
            // legacy
        if (generationOutput.size() == 1) {
            std::vector<int64_t> tokens = generationOutput[0].generated_token_ids;
            response = serializeUnaryResponse(tokenizer->decode(tokens), openAIRequest->getEndpoint(), *openAIRequest, created);
        } else {
            // Beam search only supported for unary
            std::vector<std::string> completions;
            for (GenerationOutput& out : generationOutput) {
                std::vector<int64_t> tokens = out.generated_token_ids;
                std::string completion = tokenizer->decode(tokens);
                completions.emplace_back(completion);
            }
            response = serializeUnaryResponse(completions, openAIRequest->getEndpoint(), *openAIRequest, created);
        }
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Complete unary response: {}", response);
        return StatusCode::OK;
    }



    Status inferStream(HttpPayload& request, tensorflow::serving::net_http::ServerRequestInterface* serverReaderWriter) {
        // Register resource creation time
        auto created = std::chrono::system_clock::now();
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Request body: {}", request.body);
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Request uri: {}", request.uri);
        Endpoint endpoint;
        if (request.uri == "/v3/chat/completions") {
            endpoint = Endpoint::CHAT_COMPLETIONS;
        } else if (request.uri == "/v3/completions") {
            endpoint = Endpoint::COMPLETIONS;
        } else {
            //return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
            return StatusCode::INVALID_MESSAGE_STRUCTURE;
        }

        auto openAIRequest = std::make_shared<OpenAIChatCompletionsRequest>(*request.parsedJson, endpoint);

        // TODO: Support chat scenario once atobisze adds that to CB library
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Parsing request");
        auto status = openAIRequest->parse();
        if (!status.ok())
            return status;

        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Request parsed");
        std::string finalPrompt = "";

        // LOG(INFO) << "Input prompt:" << templateApplyOutput;

        std::string prompt;
        switch (endpoint) {
        case Endpoint::CHAT_COMPLETIONS: {
            if (openAIRequest->getMessages().size() <= 0) {
                SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Returning with error");
                //return absl::Status(absl::StatusCode::kInvalidArgument, "There are no messages to apply for chat");
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            if (!applyChatTemplate(request.body, finalPrompt)) {
                SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Returning with error 2");
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            break;
        }
        case Endpoint::COMPLETIONS: {
            if (!openAIRequest->getPrompt().has_value() || !openAIRequest->getPrompt().value().size()) {
                //return absl::Status(absl::StatusCode::kInvalidArgument, "Prompt is missing");
                return StatusCode::INVALID_MESSAGE_STRUCTURE;
            }
            finalPrompt = openAIRequest->getPrompt().value();
        }
        }
        SPDLOG_LOGGER_DEBUG(llm_executor_logger, "Adding request");
        auto generationHandle = llmExecutor.pipe->add_request(
            currentRequestId++, /*to be removed from API?*/
            finalPrompt,
            openAIRequest->createGenerationConfig());

        notifyNewRequestArrived();

        auto streamer = std::make_shared<TextStreamer>(llmExecutor.pipe->get_tokenizer());
        while (generationHandle->get_status() == GenerationStatus::RUNNING || generationHandle->can_read()) {
            // Subsequent iteration
            GenerationOutputs generationOutputs = generationHandle->read();
            assert(generationOutputs.size() == 1);  // TODO: Support multiple generations
            assert(generationOutputs.begin()->second.generated_token_ids.size() == 1);
            // TODO(dkalinow): Move this logic to CB library
            int64_t token = generationOutputs.begin()->second.generated_token_ids[0];
            auto chunk = streamer->put(token);
            if (chunk.has_value()) {
                std::string response = packIntoServerSideEventMessage(
                    serializeStreamingChunk(chunk.value(), false, openAIRequest->getEndpoint(), *openAIRequest, created));
                serverReaderWriter->PartialReply(response);
                
            }
        }
        std::string response = packIntoServerSideEventMessage(serializeStreamingChunk("", true, openAIRequest->getEndpoint(), *openAIRequest, created));
        response += packIntoServerSideEventMessage("[DONE]");
        serverReaderWriter->PartialReply(response);
        return StatusCode::OK;
    }
};
}  // namespace ovms
