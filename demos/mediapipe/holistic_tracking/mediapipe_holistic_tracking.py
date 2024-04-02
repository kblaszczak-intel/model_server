#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.append("../../../../demos/common/python")

import numpy as np
import cv2

import datetime
import argparse
import os
import subprocess
import shutil
import urllib.request

import tritonclient.grpc as grpcclient

def run_command(command):
    print(command)
    if subprocess.call(command.split()) != 0:
      sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='first_input_video', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='output',
                        help='Specify output name. default: output')
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--graph_name', default='holisticTracking', help='Define graph name, must be same as is in service. default: holisticTracking',
                        dest='graph_name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')

    error = False
    args = vars(parser.parse_args())

    print("Running demo application.")
    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()
    batch_size = int(args.get('batchsize'))
    while batch_size > len(lines):
        lines += lines

    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tGraph name: {}'.format(args.get('graph_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    graph_name = args.get('graph_name')

    try:
        triton_client = grpcclient.InferenceServerClient(
            url=address,
            ssl=args['tls'],
            verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    processing_times = np.zeros((0),int)

    TIMESTAMP_PARAM_NAME = 'OVMS_MP_TIMESTAMP'

    def callback(result, error):
        if error:
            raise error
        output = result.as_numpy(output_name)
        out = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        timestamp = result.get_response().parameters[TIMESTAMP_PARAM_NAME].int64_param
        print(timestamp)
        cv2.imwrite("image_" + str(timestamp) + "_stream.jpg", out)
        
    triton_client.start_stream(callback=callback)

    for line in lines:
        inputs = []
        if not os.path.exists(line.strip()):
            print("Image does not exist: " + line.strip())
        im_cv = cv2.imread(line.strip()) 
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        print(img.shape)
        inputs.append(grpcclient.InferInput(args['input_name'], img.shape, "UINT8"))
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        
        inputs[0].set_data_from_numpy(img)
        start_time = datetime.datetime.now()
        #results = triton_client.infer(model_name=graph_name,
        #                          inputs=inputs,
        #                          outputs=outputs)
        triton_client.async_stream_infer(graph_name, inputs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        #output = results.as_numpy(output_name)
        #nu = np.array(output)

        iteration = iteration + 1
