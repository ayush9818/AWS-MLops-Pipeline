import json
import os
import argparse
import boto3
import time

sm_client = boto3.client(service_name="sagemaker", region_name="ap-southeast-1")


def create_model(model_name, ecr_image, model_uri, role, multi_model=True):
    if not multi_model:
        container = {
            "Image": ecr_image,
            "ModelDataUrl": model_uri,
            "Mode": "SingleModel",
        }
    else:
        container = {
            "Image": ecr_image,
            "ModelDataUrl": model_uri,
            "Mode": "MultiModel",
        }

    response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=[container],
    )
    print("{} Model Created".format(model_name))


def get_real_time_config(endpoint_config):
    required_fields = ['instance_type', 'instance_count', 'model_name']
    for field in required_fields:
        assert field in endpoint_config, f"{field} missing in endpoint_config"
    
    config = {
        "InstanceType": endpoint_config.get('instance_type'),
        "InitialInstanceCount": endpoint_config.get('instance_count'),
        "InitialVariantWeight": 1,
        "ModelName": endpoint_config.get('model_name'),
        "VariantName": "Variant1",
        }
    return config


def create_config(endpoint_config):
    product_config = get_real_time_config(endpoint_config)
    if "data_capture_config" in endpoint_config:
        _temp_config = endpoint_config.get("data_capture_config")
        data_capture_config = {
             'EnableCapture' : _temp_config.get("enable_capture"),
            'InitialSamplingPercentage' : _temp_config.get("sampling_percentage"), # Optional
            'DestinationS3Uri' : _temp_config.get("s3_capture_upload_path"), # Optional
            'CaptureOptions' : [{"CaptureMode": capture_mode} for capture_mode in _temp_config.get("capture_modes")]
        }
        response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config.get('config_name'),
            DataCaptureConfig=data_capture_config,
            ProductionVariants=[
                product_config
            ],
        )
    else:
        response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config.get('config_name'),
            ProductionVariants=[
                product_config
            ],
        )
    print("{} Config Created".format(endpoint_config.get('config_name')))
    


def create_endpoint(endpoint_name, config_name):
    response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=config_name
    )

    print("Endpoint Name : {}".format(endpoint_name))
    start = time.time()
    waiter = boto3.client("sagemaker", region_name="ap-southeast-1").get_waiter(
        "endpoint_in_service"
    )
    print("Waiting for endpoint to create...")
    waiter.wait(EndpointName=endpoint_name)
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint Status: {}".format(resp["EndpointStatus"]))
    print("Endpoint Creation Time : {}".format(time.time() - start))


def delete_resources(endpoint_name, config_name, model_name, aws_region):
    sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint Deleted : {endpoint_name}")
    except Exception as e:
        print(e)

    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Endpoint Configuration Deleted : {config_name}")
    except Exception as e:
        print(e)

    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"Endpoint Configuration Deleted : {model_name}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="endpoint configuration file path")
    parser.add_argument(
        "--action",
        type=str,
        help="Supported Actions : create_endpoint, delete_endpoint",
    )
    parser.add_argument(
        "--endpoint-type",
        type=str,
        help="Supported Endpoint Types : real-time-endpoint, multi-model-endpoint"
    )
    args = parser.parse_args()

    config_path = args.cfg
    action = args.action
    endpoint_type = args.endpoint_type

    assert action in [
        "delete_endpoint",
        "create_endpoint",
    ], f"Supported Actions are : create_endpoint and delete_endpoint"

    assert endpoint_type in [
        "real-time-endpoint",
        "multi-model-endpoint",
    ], f"Supported Endpoint Types are : real-time-endpoint and multi-model-endpoint"

    assert os.path.exists(config_path), f"{config_path} does not exist"

    endpoint_master_config = json.load(open(config_path))
    endpoint_config = endpoint_master_config.get(endpoint_type)

    if action == "create_endpoint":
        if endpoint_type in ["real-time-endpoint"]:
            create_model(
                model_name=endpoint_config.get("model_name"),
                ecr_image=endpoint_config.get("container_uri"),
                model_uri=endpoint_config.get("model_uri"),
                role=endpoint_config.get("iam_role"),
                multi_model=False
            )
        else:
            create_model(
                model_name=endpoint_config.get("model_name"),
                ecr_image=endpoint_config.get("container_uri"),
                model_uri=endpoint_config.get("model_uri"),
                role=endpoint_config.get("iam_role"),
                multi_model=True
            )
            
        create_config(endpoint_config)
            
        create_endpoint(
            endpoint_name=endpoint_config.get("endpoint_name"),
            config_name=endpoint_config.get("config_name"),
        )

    if action == "delete_endpoint":
        delete_resources(
            endpoint_name=endpoint_config.get("endpoint_name"),
            config_name=endpoint_config.get("config_name"),
            model_name=endpoint_config.get("model_name"),
            aws_region=endpoint_config.get("region"),
        )
