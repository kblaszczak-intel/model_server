//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

namespace mediapipe {

/*
    Adds 1 to all bytes input tensor.
*/
class StreamingTestCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "StreamingTestCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingTestCalculator::Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingTestCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingTestCalculator::Process";
        ov::Tensor input = cc->Inputs().Index(0).Get<ov::Tensor>();
        ov::Tensor output(input.get_element_type(), input.get_shape());
        for (size_t i = 0; i < input.get_byte_size() / 4; i++) {
            ((float*)(output.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
        }
        cc->Outputs().Index(0).Add(new ov::Tensor(output), cc->InputTimestamp());
        return absl::OkStatus();
    }
};

/*
    Adds 1 to all bytes to one of non empty input tensors
    Produces 2 the same copies of added tensor
*/
class StreamingCycleTestCalculator : public CalculatorBase {
    int cycle_iteration = 0;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "StreamingCycleTestCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        cc->Inputs().Index(1).Set<ov::Tensor>();   // signal
        cc->Outputs().Index(1).Set<ov::Tensor>();  // signal
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingCycleTestCalculator::Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingCycleTestCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingCycleTestCalculator::Process";
        if (++cycle_iteration > 3) {
            return absl::OkStatus();
        }
        ov::Tensor input = cc->Inputs().Index(0).IsEmpty() ? cc->Inputs().Index(1).Get<ov::Tensor>() : cc->Inputs().Index(0).Get<ov::Tensor>();
        ov::Tensor output1(input.get_element_type(), input.get_shape());
        ov::Tensor output2(input.get_element_type(), input.get_shape());
        for (size_t i = 0; i < input.get_byte_size() / 4; i++) {
            ((float*)(output1.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
            ((float*)(output2.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
        }
        cc->Outputs().Index(0).Add(new ov::Tensor(output1), Timestamp(cycle_iteration));  // TODO: ?
        cc->Outputs().Index(1).Add(new ov::Tensor(output2), Timestamp(cycle_iteration));  // TODO: ?
        return absl::OkStatus();
    }
};

/*
    3 inputs 3 outputs
    Index 0 +1
    Index 1 +2
    Index 2 +3
*/
class StreamingMultiInputsOutputsTestCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Inputs().Index(1).Set<ov::Tensor>();
        cc->Inputs().Index(2).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(1).Set<ov::Tensor>();
        cc->Outputs().Index(2).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Process";
        ov::Tensor input1 = cc->Inputs().Index(0).Get<ov::Tensor>();
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Process";
        ov::Tensor input2 = cc->Inputs().Index(1).Get<ov::Tensor>();
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Process";
        ov::Tensor input3 = cc->Inputs().Index(2).Get<ov::Tensor>();
        LOG(INFO) << "StreamingMultiInputsOutputsTestCalculator::Process";
        ov::Tensor output1(input1.get_element_type(), input1.get_shape());
        ov::Tensor output2(input2.get_element_type(), input2.get_shape());
        ov::Tensor output3(input3.get_element_type(), input3.get_shape());
        for (size_t i = 0; i < input1.get_byte_size() / 4; i++) {
            ((float*)(output1.data()))[i] = ((float*)(input1.data()))[i] + 1.0f;
            ((float*)(output2.data()))[i] = ((float*)(input2.data()))[i] + 1.0f;
            ((float*)(output3.data()))[i] = ((float*)(input3.data()))[i] + 1.0f;
        }
        cc->Outputs().Index(0).Add(new ov::Tensor(output1), cc->InputTimestamp());  // TODO: ?
        cc->Outputs().Index(1).Add(new ov::Tensor(output2), cc->InputTimestamp());  // TODO: ?
        cc->Outputs().Index(2).Add(new ov::Tensor(output3), cc->InputTimestamp());  // TODO: ?
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(StreamingTestCalculator);
REGISTER_CALCULATOR(StreamingCycleTestCalculator);
REGISTER_CALCULATOR(StreamingMultiInputsOutputsTestCalculator);
}  // namespace mediapipe
