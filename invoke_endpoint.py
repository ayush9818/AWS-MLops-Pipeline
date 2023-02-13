import json
import os
import argparse
import boto3
import time


def invoke_endpoint(inference_config):
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=inference_config.get('region'))
    endpoint_name=inference_config.get('endpoint_name')
    payload = json.load(open(inference_config.get('payload_path')))
    response = sagemaker_runtime.invoke_endpoint(
                            EndpointName=endpoint_name, 
                            Body=json.dumps(payload)
                            )
    result = json.loads(response['Body'].read().decode('utf-8'))
    print(result)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help='inference configuration file path')
    args = parser.parse_args()
    
    config_path = args.cfg
    assert os.path.exists(config_path), f"{config_path} does not exist"
    
    inference_config = json.load(open(config_path))
    invoke_endpoint(inference_config)
