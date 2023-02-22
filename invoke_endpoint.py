import json
import os
import argparse
import boto3
import sagemaker
import urllib, time
from botocore.exceptions import ClientError

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

def upload_file_on_s3(bucket_name, upload_dir, input_location):
    sm_session = sagemaker.session.Session()
    return sm_session.upload_data(
        input_location,
        bucket=bucket_name,
        key_prefix=upload_dir,
        extra_args={"ContentType": "application/json"},
    )

def get_output(output_location):
    sm_session = sagemaker.session.Session()
    output_url = urllib.parse.urlparse(output_location)
    bucket = output_url.netloc
    key = output_url.path[1:]
    while True:
        try:
            return sm_session.read_s3_file(bucket=output_url.netloc, key_prefix=output_url.path[1:])
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print("waiting for output...")
                time.sleep(2)
                continue
            raise

def invoke_async_endpoint(inference_config):
    sm_runtime = boto3.client("sagemaker-runtime")
    input_s3_location = upload_file_on_s3(input_location=inference_config.get("payload_path"),
                                      bucket_name=inference_config.get('bucket_name'),
                                      upload_dir=inference_config.get('upload_dir')
                                     )
    print(input_s3_location)
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=inference_config.get('endpoint_name'),
        InputLocation=input_s3_location
    )
    output_location = response["OutputLocation"]
    print(f"OutputLocation: {output_location}")
    output = get_output(output_location)
    print(f"Output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="inference configuration file path")
    parser.add_argument(
        "--endpoint-type",
        type=str,
        help="Supported Endpoint Types : real-time-endpoint, multi-model-endpoint,serverless-endpoint, async-endpoint",
    )
    args = parser.parse_args()

    config_path = args.cfg
    endpoint_type = args.endpoint_type

    assert endpoint_type in [
        "real-time-endpoint",
        "multi-model-endpoint",
        "serverless-endpoint",
        "async-endpoint"
    ], f"Supported Endpoint Types are : real-time and multi-model"
    assert os.path.exists(config_path), f"{config_path} does not exist"

    inference_config = json.load(open(config_path)).get(endpoint_type)

    if endpoint_type == "real-time-endpoint":
        invoke_real_time_endpoint(inference_config)
    elif endpoint_type == "multi-model-endpoint":
        invoke_multi_model_endpoint(inference_config)
    elif endpoint_type == "async-endpoint":
        invoke_async_endpoint(inference_config)
    else:
        invoke_serverless_endpoint(inference_config)
