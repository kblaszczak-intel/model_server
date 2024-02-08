#
# Copyright (c) 2024 Intel Corporation
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
sys.path.append("../../common/python")
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
import tritonclient.http as httpclient
import argparse
import datetime
import json
import numpy as np
from client_utils import print_statistics
from urllib.request import urlretrieve
from pathlib import Path
import os
import grpc
import time
import cv2

parser = argparse.ArgumentParser(description='Client for clip example')

parser.add_argument('--timeout', required=False, default='15',
                    help='Specify timeout to wait for models readiness on the server in seconds. default 15 seconds.')
parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--input_labels', required=False, default="cat,dog,wolf,tiger,man,horse,frog,tree,house,computer",
                    help="Specify input_labels to the CLIP model. default:cat,dog,wolf,tiger,man,horse,frog,tree,house,computer")
parser.add_argument('--image_url', required=False, default='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg',
                    help='Specify image_url to send to the CLIP model. default:https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg')
parser.add_argument('--iterations', default=1,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 1 ',
                        dest='iterations', type=int)
parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--server_cert', required=False, help='Path to server certificate', default=None)
parser.add_argument('--client_cert', required=False, help='Path to client certificate', default=None)
parser.add_argument('--client_key', required=False, help='Path to client key', default=None)
parser.add_argument('--http_address',required=False, default='localhost',  help='Specify url to http service. default:localhost')
parser.add_argument('--http_port',required=False, default=8000, help='Specify port to http service. default: 8000')
parser.add_argument('--binary_data', default=False, action='store_true', help='Send input data in binary format', dest='binary_data')

args = vars(parser.parse_args())

iterations = args.get('iterations')
iteration = 0

address = "{}:{}".format(args['http_address'],args['http_port'])
if args['tls']:
        ssl_options = {
            'keyfile':args['client_key'],
            'cert_file':args['client_cert'],
            'ca_certs':args['server_cert']
        }
else:
    ssl_options = None

triton_client = httpclient.InferenceServerClient(
                url=address,
                ssl=args['tls'],
                ssl_options=ssl_options,
                verbose=False)

image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")

input_name = image_url.split("/")[-1]
sample_path = Path(os.path.join("data", input_name))
if not os.path.exists(sample_path):
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        image_url,
        sample_path,
    )

img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)

input_labels_array = [args['input_labels']]
input_labels = args['input_labels'].split(",")
print(f"Using input_labels:\n{input_labels}\n")

image_data = []
with open(sample_path, "rb") as f:
    image_data.append(f.read())

#nmpy = np.array(image_data , dtype=np.object_)
npydata = np.array(image_data, dtype=np.bytes_)
npylabelsdata = np.array(input_labels_array, dtype=np.bytes_)
inputs = []
inputs.append(httpclient.InferInput('image', [len(npydata)], "BYTES"))
inputs[0].set_data_from_numpy(npydata)

inputs.append(httpclient.InferInput('input_labels', [len(npylabelsdata)], "BYTES"))
inputs[1].set_data_from_numpy(npylabelsdata)

parameters = {
        "binary_data_size" : 16
      }

processing_times = []
for iteration in range(iterations):
    outputs = []
    print(f"Iteration {iteration}")
    start_time = datetime.datetime.now()

    model_name = "python_model"
    #results = requests.post(f'http://{rest_address}:{rest_port}/v2/models/{model_name}/infer', data=data_json, timeout=15)

    results = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                parameters=parameters)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times.append(int(duration))

    result_dict = json.loads(results.text)

    print(f"Detection:\n{results.as_numpy['output_label'].tobytes().decode()}\n")

print_statistics(np.array(processing_times,int), batch_size = 1)