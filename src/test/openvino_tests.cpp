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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include "ocl_utils.hpp"
#include "c_api_test_utils.hpp"
#include "test_utils.hpp"

using namespace ov;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

cl_context get_cl_context(cl_platform_id& platformId, cl_device_id& deviceId) {
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms\n";
        throw 1;
    }
    // extract 1st platform from numPlatforms
    clGetPlatformIDs(1, &platformId, nullptr);
    cl_uint numDevices = 0;
    // query how many devices there are
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting number of devices\n";
        throw 1;
    }
    if (0 == numDevices) {
        std::cerr << "There is no available devices\n";
        throw 1;
    }
    cl_uint numberOfDevicesInContext = 1;
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numberOfDevicesInContext, &deviceId, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting GPU deviceId\n";
        throw 1;
    }
    // since we only use 1 device we can use address of deviceId
    cl_context openCLCContext = clCreateContext(nullptr, numberOfDevicesInContext, &deviceId, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context\n";
        throw 1;
    }
    return openCLCContext;
}

TEST(OpenVINO, ExtractContextFromModel) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
        ov::auto_batch_timeout(0)};
    cl_platform_id platformId;
    cl_device_id deviceId;
    auto nonused = get_cl_context(platformId, deviceId);
    auto compiledModel = core.compile_model(model, "GPU", config);
    auto gpu_context = compiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctxFromModel = gpu_context.get();
    cl::Context openCLCppContext(ctxFromModel);
    cl_int err;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    auto inputOVOCLBufferTensor = gpu_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = gpu_context.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    cl::Device device(deviceId);
    auto queue = cl::CommandQueue(openCLCppContext, device);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* buffer_out = out.data();
    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, outputByteSize, buffer_out);
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

TEST(OpenVINO, LoadModelWithPrecreatedContext) {
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    auto input = model->get_parameters().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto outputByteSize = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Device device(deviceId);
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
    auto compiledModel = core.compile_model(model, remote_context);
    // now we create buffers
    cl::Context openCLCppContext(openCLCContext);
    cl_int err;
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err);
    // create tensors and perform inference
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto inputOVOCLBufferTensor = remote_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
    auto outputOVOCLBufferTensor = remote_context.create_tensor(output->get_element_type(), output->get_shape(), openCLCppOutputBuffer);
    // we will put data into input buffer
    std::vector<float> in(10, 0.1);
    void* inputBufferData = in.data();
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    auto inferRequest = compiledModel.create_infer_request();
    inferRequest.set_tensor(input, inputOVOCLBufferTensor);
    inferRequest.set_tensor(output, outputOVOCLBufferTensor);
    inferRequest.infer();
    std::vector<float> out(10);
    void* buffer_out = out.data();
    queue.enqueueReadBuffer(openCLCppOutputBuffer, /*blocking*/ true, 0, outputByteSize, buffer_out);
    for (size_t i = 0; i < inputByteSize / sizeof(float); ++i) {
        // different precision on GPU vs CPU
        EXPECT_NEAR(in[i] + 1, out[i], 0.0004) << "i:" << i;
    }
}

TEST(OpenVINO, SetTensorTest) {
    size_t tSize = 10;
    int iterations = 1000;
    std::vector<size_t> sizeSet{10, 10 * 10, 10 * 100, 10 * 1000, 10 * 10000, 10 * 100000, 10 * 1000000};
    // load model
    Core core;
    auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
    const std::string inputName{"b"};
    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);
    auto inputByteSize = ov::shape_size(input->get_shape());
    auto outputByteSize = ov::shape_size(output->get_shape());
    // we need byte size not no of elements
    inputByteSize *= sizeof(float);
    outputByteSize *= sizeof(float);
    enum {
        // DEV_CONTEXT_SCENARIO
        CPU_COPY, // regular OVMS scenario
        CPU_SET, // set output tensors to avoid copy
        GPU_OV_COPY_OV, // regular GPU OVMS scenario
        GPU_OV_SET_OV, // set regular ov tensors and use gpu for inference
        GPU_OCL_COPY, // model loadded with OCL use OV tensors on input and still copy output
        GPU_OCL_SET_OV, // set regular ov tensors and use gpu with passed in context for inference
        GPU_OCL_SET_OCL_IN_AND_OV_OUT, // set ocl tensor on input and  ov tensors on output and use gpu with passed in context for inference
        GPU_OCL_SET_OCL, // set OCL tensors and use gpu with passed in context for inference
        GPU_OCL_DIFF_CONTEXT_INPUT_COPY, // use OCL tensors on input but use different context
        GPU_OV_SET_OCL_DIFF_CONTEXT, // set OCL tensors and use gpu for inference but model with default OV context
        GPU_OV_SET_OCL_SAME_CONTEXT, // set OCL tensors with the default OV context with gpu
    };
    std::unordered_map<int, std::unordered_map<int, double>> times;
    for (auto tSize : sizeSet) {
        SPDLOG_ERROR("Performing tests for dummy shape (1,{}) ....", tSize);
        auto sizeStart = std::chrono::high_resolution_clock::now();
        ov::element::Type_t dtype = ov::element::Type_t::f32;
        ov::Shape ovShape;
        ovShape.emplace_back(1);
        ovShape.emplace_back(tSize);
        std::map<std::string, ov::PartialShape> inputShapes;
        inputShapes[inputName] = ovShape;
        model->reshape(inputShapes);
        auto gpuCompiledModel = core.compile_model(model, "GPU");
        auto gpuInferRequest = gpuCompiledModel.create_infer_request();
        auto cpuCompiledModel = core.compile_model(model, "CPU");
        auto cpuInferRequest = cpuCompiledModel.create_infer_request();
        // prepare ov::Tensor data
        std::vector<ov::Tensor> inputOvTensors, outputOvTensors;
        inputOvTensors.emplace_back(dtype, ovShape);
        inputOvTensors.emplace_back(dtype, ovShape);
        outputOvTensors.emplace_back(dtype, ovShape);
        outputOvTensors.emplace_back(dtype, ovShape);
        auto inputByteSize = tSize;
        auto outputByteSize = tSize;
        inputByteSize *= sizeof(float);
        outputByteSize *= sizeof(float);

        cl_int err;
        std::vector<float> outputData(tSize, 0);
        {  // GPU_OCL_COPY model loaded with OCL context, using ov::Tensors on input & output (copy)
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            auto start = std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                oclInferRequest.infer();
                const auto& outTensor = oclInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            SPDLOG_ERROR("finished GPU_OV_COPY_OV");
            times[GPU_OCL_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        }
        if (false /*not working*/) {  // GPU_OCL_DIFF_CONTEXT_INPUT_COPY model loaded with OCL context using OCL tensors on input from different context, copying output
            // illegal [GPU] trying to reinterpret buffer allocated by a different engine
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl_context openCLCContextDifferent = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            cl::Context openCLCppContextDifferent(openCLCContextDifferent);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto ovWrappedOCLContextDifferent = ov::intel_gpu::ocl::ClContext(core, openCLCContextDifferent, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextDifferent, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextDifferent.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextDifferent.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));

            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.infer();
                const auto& outTensor = oclInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OCL_DIFF_CONTEXT_INPUT_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_DIFF_CONTEXT_INPUT_COPY");
        }
        {  // GPU_OCL_SET_OCL using model loaded with OCL & tensor from the same context on both input & output
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));

            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.set_tensor(output, outputs[i % 2]);
                oclInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OCL][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OCL");
        }
        // TODO FIXME
        {  // GPU_OCL_SET_OCL_IN_AND_OV_OUT using model loaded with OCL & tensor on input from the same context.Output using ov;:Tensor & copy
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputs[i % 2]);
                oclInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                oclInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OCL_IN_AND_OV_OUT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OCL_IN_AND_OV_OUT");
        }
        {  // GPU_OCL_SET_OV model loaded on gpu with both outpu & input being ov::Tensor
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            auto oclCompiledModel = core.compile_model(model, ovWrappedOCLContext);
            auto oclInferRequest = oclCompiledModel.create_infer_request();
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                oclInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                oclInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                oclInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OCL_SET_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OCL_SET_OV");
        }
        {
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                gpuInferRequest.infer();
                const auto& outTensor = gpuInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OV_COPY_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_COPY_OV");
            // GPU set input & output
        }
        { // GPU_OV_SET_OV inference with ov::Tensors but output is set as well
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                gpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OV][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_TRACE("finished GPU_OV_SET_OV");
        }
        { // CPU_COPY inference with ov::Tensors - current (2024.1) OVMS flow with cpu
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                cpuInferRequest.infer();
                const auto& outTensor = cpuInferRequest.get_tensor(output);
                std::memcpy(outputData.data(), outTensor.data(), outputByteSize);
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[CPU_COPY][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished CPU_COPY");
        }
        { // CPU_SET inference with ov::Tensors but output is set as well
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                cpuInferRequest.set_tensor(input, inputOvTensors[i % 2]);
                cpuInferRequest.set_tensor(output, outputOvTensors[i % 2]);
                cpuInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[CPU_SET][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished CPU_SET");
        }
        if (false) {  // GPU_OV_SET_OCL_DIFF_CONTEXT model loaded with ov context and different ocl context used to create ocl tensors
            // illegal [GPU] trying to reinterpret buffer allocated by a different engine
            cl_platform_id platformId;
            cl_device_id deviceId;
            cl_context openCLCContext = get_cl_context(platformId, deviceId);
            cl::Context openCLCppContext(openCLCContext);
            auto ovWrappedOCLContext = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContext, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContext.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContext.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_DIFF_CONTEXT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_DIFF_CONTEXT");
        }
        {  // GPU_OV_SET_OCL_SAME_CONTEXT load model with target device and use context from model to create tensors
            auto ovWrappedOCLContextFromModel = gpuCompiledModel.get_context().as<ov::intel_gpu::ocl::ClContext>();
            cl_context openCLCContextFromModel = ovWrappedOCLContextFromModel.get();
            cl::Context openCLCppContextFromModel(openCLCContextFromModel);
            // prepare tensors
            std::vector<cl::Buffer> inputsBuffers, outputsBuffers;
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            inputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, inputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            outputsBuffers.emplace_back(cl::Buffer(openCLCppContextFromModel, CL_MEM_READ_WRITE, outputByteSize, NULL, &err));
            std::vector<ov::intel_gpu::ocl::ClBufferTensor> inputs, outputs;
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[0]));
            inputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(input->get_element_type(), input->get_shape(), inputsBuffers[1]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[0]));
            outputs.emplace_back(ovWrappedOCLContextFromModel.create_tensor(output->get_element_type(), output->get_shape(), outputsBuffers[1]));
            auto start =std::chrono::high_resolution_clock::now();
            for (auto i = 0; i < iterations; ++i) {
                gpuInferRequest.set_tensor(input, inputs[i % 2]);
                gpuInferRequest.set_tensor(output, outputs[i % 2]);
                gpuInferRequest.infer();
            }
            auto stop =std::chrono::high_resolution_clock::now();
            times[GPU_OV_SET_OCL_SAME_CONTEXT][tSize] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            SPDLOG_ERROR("finished GPU_OV_SET_OCL_SAME_CONTEXT");
        }
        auto sizeStop = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(sizeStop - sizeStart).count() / 1000000.0;
        SPDLOG_ERROR("For size: {:8d} inferences all took {:03.5f} seconds. Next inferences will take probably ~x10 longer ...", tSize, totalTime);
    }
    std::cout << std::right;
    for (auto s : {"CPU_COPY", "CPU_SET", "GPU_OV_COPY_OV", "GPU_OV_SET_OV", "GPU_OCL_COPY", "GPU_OCL_SET_OV", "GPU_OCL_SET_OCL_IN_AND_OV_OUT", "GPU_OCL_SET_OCL", /*"GPU_OCL_DIFF_CONTEXT_INPUT_COPY", "GPU_OV_SET_OCL_DIFF_CONTEXT",*/ "GPU_OV_SET_OCL_SAME_CONTEXT"}) {
        std::cout << s << "[MePS]"
            << "\t\t";
    }
    std::cout << std::endl;
    for (auto s : sizeSet) {
        for (auto t : {CPU_COPY, CPU_SET, GPU_OV_COPY_OV, GPU_OV_SET_OV, GPU_OCL_COPY, GPU_OCL_SET_OV, GPU_OCL_SET_OCL_IN_AND_OV_OUT, GPU_OCL_SET_OCL, /*GPU_OCL_DIFF_CONTEXT_INPUT_COPY, GPU_OV_SET_OCL_DIFF_CONTEXT,*/ GPU_OV_SET_OCL_SAME_CONTEXT}) {
            double fps = iterations / (times[t][s] / 1000.);
            std::cout << "" << fps * s << " \t\t ";
        }
        std::cout << std::endl;
    }
}

#include "../ocl_utils.hpp"

TEST(CAPINonCopy, Flow) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context openCLCContext = get_cl_context(platformId, deviceId);
    cl::Context openCLCppContext(openCLCContext);
    cl::Device device(deviceId);
    cl_command_queue_properties oclQueueProperties = false ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : CL_NONE;
    auto queue = cl::CommandQueue(openCLCppContext, device, oclQueueProperties);
    // create OpenCL buffers
    std::vector<float> in(10, 42);
    void* inputBufferData = in.data();
    size_t inputByteSize = sizeof(float) * in.size();
    cl_int err;  // TODO not ignore
    cl::Buffer openCLCppInputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    cl::Buffer openCLCppOutputBuffer(openCLCppContext, CL_MEM_READ_WRITE, inputByteSize, NULL, &err);
    queue.enqueueWriteBuffer(openCLCppInputBuffer, /*blocking*/ true, 0, inputByteSize, inputBufferData);
    // start CAPI server
    // TODO load model with passed in context
    std::string port = "9000";
    randomizePort(port);
    OVMS_ServerSettings* serverSettings = nullptr;
    OVMS_ModelsSettings* modelsSettings = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsNew(&serverSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsNew(&modelsSettings));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerSettingsSetGrpcPort(serverSettings, std::stoi(port)));
    ASSERT_CAPI_STATUS_NULL(OVMS_ModelsSettingsSetConfigPath(modelsSettings, "/ovms/src/test/c_api/config_gpu_dummy.json"));
    OVMS_Server* cserver = nullptr;
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerNew(&cserver));
    ASSERT_CAPI_STATUS_NULL(OVMS_ServerStartFromConfigurationFile(cserver, serverSettings, modelsSettings));
    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestNew(&request, cserver, "dummy", 1));
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size()));
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t notUsedNum = 0;
    auto bareOpenCLCMemory = openCLCppInputBuffer.get();
    //ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(bareOpenCLCMemory), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1)); // device id ?? TODO
    SPDLOG_ERROR("ERa addr:{}", (void*)&openCLCppInputBuffer);
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(&openCLCppInputBuffer), inputByteSize, OVMS_BUFFERTYPE_OPENCL, 1));  // device id ?? TODO
                                                                                                                                                                                            // verify response
    OVMS_InferenceResponse* response = nullptr;
    //ASSERT_CAPI_STATUS_NULL(OVMS_Inference(cserver, request, &response));
    EXPECT_EQ(nullptr, OVMS_Inference(cserver, request, &response));
    {
        ov::Core core;
        auto model = core.read_model("/ovms/src/test/dummy/1/dummy.xml");
        auto input = model->get_parameters().at(0);
        auto inputByteSize = ov::shape_size(input->get_shape());
        auto output = model->get_results().at(0);
        auto outputByteSize = ov::shape_size(output->get_shape());
        inputByteSize *= sizeof(float);
        outputByteSize *= sizeof(float);
        ov::AnyMap config = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
            ov::auto_batch_timeout(0)};
        cl_platform_id platformId;
        cl_device_id deviceId;
        auto remote_context = ov::intel_gpu::ocl::ClContext(core, openCLCContext, 0);
        auto compiledModel = core.compile_model(model, remote_context);
        auto inferRequest = compiledModel.create_infer_request();

        auto inputOVOCLBufferTensor = remote_context.create_tensor(input->get_element_type(), input->get_shape(), openCLCppInputBuffer);
        inferRequest.set_tensor(input, inputOVOCLBufferTensor);
        inferRequest.start_async();
        inferRequest.wait();
    }
    uint32_t outputCount = 42;
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutputCount(response, &outputCount));
    ASSERT_EQ(outputCount, 1);
    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const int64_t* shape{nullptr};
    size_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t capiDeviceId = 42;
    const char* outputName{nullptr};
    ASSERT_CAPI_STATUS_NULL(OVMS_InferenceResponseOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &capiDeviceId));
    ASSERT_EQ(std::string(DUMMY_MODEL_OUTPUT_NAME), outputName);
    EXPECT_EQ(datatype, OVMS_DATATYPE_FP32);
    EXPECT_EQ(dimCount, 2);
    EXPECT_EQ(bufferType, OVMS_BUFFERTYPE_CPU);  // TODO ?
    EXPECT_EQ(capiDeviceId, 0);                  // TODO?
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
        EXPECT_EQ(DUMMY_MODEL_SHAPE[i], shape[i]) << "Different at:" << i << " place.";
    }
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    ASSERT_EQ(bytesize, sizeof(float) * DUMMY_MODEL_INPUT_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(in[i] + 1, outputData[i]) << "Different at:" << i << " place.";
    }
    // TODO
    // infer
    // TODO need to pass context into MP ??
}
#pragma GCC diagnostic pop
