import json
import os
import argparse
import boto3


def invoke_real_time_endpoint(inference_config):
    sagemaker_runtime = boto3.client(
        "sagemaker-runtime", region_name=inference_config.get("region")
    )
    endpoint_name = inference_config.get("endpoint_name")
    payload = json.load(open(inference_config.get("payload_path")))
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    print(result)


def invoke_multi_model_endpoint(inference_config):
    sagemaker_runtime = boto3.client(
        "sagemaker-runtime", region_name=inference_config.get("region")
    )
    endpoint_name = inference_config.get("endpoint_name")
    payload = json.load(open(inference_config.get("payload_path")))
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        TargetModel=inference_config.get("target_model"),
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    print(result)
    
def invoke_serverless_endpoint(inference_config):
    sagemaker_runtime = boto3.client(
        "sagemaker-runtime", region_name=inference_config.get("region")
    )
    endpoint_name = inference_config.get("endpoint_name")
    payload = json.load(open(inference_config.get("payload_path")))
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="inference configuration file path")
    parser.add_argument(
        "--endpoint-type",
        type=str,
        help="Supported Endpoint Types : real-time-endpoint, multi-model-endpoint,serverless-endpoint",
    )
    args = parser.parse_args()

    config_path = args.cfg
    endpoint_type = args.endpoint_type

    assert endpoint_type in [
        "real-time-endpoint",
        "multi-model-endpoint",
        "serverless-endpoint"
    ], f"Supported Endpoint Types are : real-time and multi-model"
    assert os.path.exists(config_path), f"{config_path} does not exist"

    inference_config = json.load(open(config_path)).get(endpoint_type)

    if endpoint_type == "real-time-endpoint":
        invoke_real_time_endpoint(inference_config)
    elif endpoint_type == "multi-model-endpoint":
        invoke_multi_model_endpoint(inference_config)
    else:
        invoke_serverless_endpoint(inference_config)
