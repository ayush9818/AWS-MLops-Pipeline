import json
import os
import argparse
import boto3

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html : Refer to the documentation for more params

def invoke_real_time_endpoint(inference_config):
    endpoint_name=inference_config.get('endpoint_name')
    predictor = Predictor(endpoint_name=endpoint_name,
                        serializer=JSONSerializer(),
                        deserializer=JSONDeserializer()
                        )
    payload = json.load(open(inference_config.get("payload_path")))
    result = predictor.predict(payload)
    print(result)

def invoke_multi_time_endpoint(inference_config):
    endpoint_name=inference_config.get('endpoint_name')
    predictor = Predictor(endpoint_name=endpoint_name,
                        serializer=JSONSerializer(),
                        deserializer=JSONDeserializer()
                        )
    payload = json.load(open(inference_config.get("payload_path")))
    result = predictor.predict(payload,target_model=inference_config.get("target_model"))
    print(result)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="inference configuration file path")
    parser.add_argument(
        "--endpoint-type",
        type=str,
        help="Supported Endpoint Types : real-time-endpoint, multi-model-endpoint"
    )
    #multi-model-endpoint","serverless-endpoint",
    #TODO : Need to Test for Multi-Model Endpoint
    args = parser.parse_args()

    config_path = args.cfg
    endpoint_type = args.endpoint_type

    assert endpoint_type in [
        "real-time-endpoint",
        "multi-model-endpoint",
        #"serverless-endpoint"
    ], f"Supported Endpoint Types are : real-time and multi-model"
    assert os.path.exists(config_path), f"{config_path} does not exist"

    inference_config = json.load(open(config_path)).get(endpoint_type)

    if endpoint_type == "real-time-endpoint":
        invoke_real_time_endpoint(inference_config)
    elif endpoint_type == "multi-model-endpoint":
        invoke_multi_time_endpoint(inference_config)
