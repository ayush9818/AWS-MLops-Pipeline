import json
import os
import argparse
import boto3
import time

sm_client = boto3.client(service_name="sagemaker", region_name="ap-southeast-1")


def create_model(model_name, ecr_image, model_uri, role, multi_model=True):
    """
    This will create a model object for the endpoint. The model endpoint basically contains following attributes
     - model_name : Name of the model object
     - ecr_image : ECR Image to be used for Inference
     - model_uri : s3 directory containing all the model artifacts in case of multi-model otherwise s3 path of model artifacts
     - Mode : SingleModel | MultiModel
    additional parameters :
        - role : iam role to be used
        - multi_model : Boolean to indicate whether to use multi-model or single model

    To change the default environment variables for async endpoint for timeouts and request/response size
    edit the container :
        container = {
            "Image": ecr_image,
            "ModelDataUrl": model_uri,
            "Mode": "SingleModel",
            'Environment': {
                'TS_MAX_REQUEST_SIZE': '100000000',
                'TS_MAX_RESPONSE_SIZE': '100000000',
                'TS_DEFAULT_RESPONSE_TIMEOUT': '1000'
            }
        }
    """
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
    """
    Creates Product configuration for real-time enpoint
    """
    required_fields = ["instance_type", "instance_count", "model_name"]
    for field in required_fields:
        assert field in endpoint_config, f"{field} missing in endpoint_config"

    config = {
        "InstanceType": endpoint_config.get("instance_type"),
        "InitialInstanceCount": endpoint_config.get("instance_count"),
        "InitialVariantWeight": 1,
        "ModelName": endpoint_config.get("model_name"),
        "VariantName": "Variant1",
    }
    return config


def get_serverless_config(endpoint_config):
    """
    Creates Product configuration for serverless endpoint
    """
    required_fields = ["model_name", "memory_size", "max_concurrency"]
    for field in required_fields:
        assert field in endpoint_config, f"{field} missing in endpoint_config"

    config = {
        "ModelName": endpoint_config.get("model_name"),
        "VariantName": "AllTraffic",
        "ServerlessConfig": {
            "MemorySizeInMB": endpoint_config.get("memory_size"),
            "MaxConcurrency": endpoint_config.get("max_concurrency"),
        },
    }
    return config


def get_data_capture_config(endpoint_config):
    """
    Create a data capture config
    """
    required_fields = [
        "enable_capture",
        "sampling_percentage",
        "s3_capture_upload_path",
        "capture_modes",
    ]
    _temp_config = endpoint_config.get("data_capture_config")
    for field in required_fields:
        assert field in _temp_config, f"{field} missing in data_capture_config"
    data_capture_config = {
        "EnableCapture": _temp_config.get("enable_capture"),
        "InitialSamplingPercentage": _temp_config.get("sampling_percentage"),
        "DestinationS3Uri": _temp_config.get("s3_capture_upload_path"),
        "CaptureOptions": [
            {"CaptureMode": capture_mode}
            for capture_mode in _temp_config.get("capture_modes")
        ],
    }
    return data_capture_config


def create_config_util(
    endpoint_config, product_config, async_config, data_capture_config
):
    if data_capture_config is not None:
        """
        If data_capture_config is not None, It is assumed that it is only real-time-endpoint.
        So, not checking for async_config
        """
        print(f"Creating Config with data_capture enabled")
        response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config.get("config_name"),
            DataCaptureConfig=data_capture_config,
            ProductionVariants=[product_config],
        )
        return response
    else:
        """
        if data_capture_config is None, then it can real-time, multi-model, serverless or async endpoint
        So, we need to check for aysnc_config, endpoint-config will vary for async and will be same for rest
        """
        if async_config is None:
            response = sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_config.get("config_name"),
                ProductionVariants=[product_config],
            )
            return response
        else:
            print(f"Using Async Endpoint Configuration")
            response = sm_client.create_endpoint_config(
                EndpointConfigName=endpoint_config.get("config_name"),
                ProductionVariants=[product_config],
                AsyncInferenceConfig=async_config,
            )
            return response


def get_async_config(endpoint_config):
    """
    This function will create a async configuration for async endpoint

    For now, not configuring SNS Topic. Added for TODO
    TODO : Configure SNS Success and Error Topic
    The output config will look the following if we include SNS Configurations as well

    "OutputConfig": {
            "S3OutputPath": f"s3://{s3_bucket}/{bucket_prefix}/output",
            "NotificationConfig": {
                "SuccessTopic": "arn:aws:sns:aws-region:account-id:topic-name",
                "ErrorTopic": "arn:aws:sns:aws-region:account-id:topic-name",
            }
        }
    """
    async_config = {
        "OutputConfig": {
            # Location to upload response outputs when no location is provided in the request.
            "S3OutputPath": endpoint_config.get("s3_output_path")
        },
        "ClientConfig": {
            # (Optional) Specify the max number of inflight invocations per instance
            # If no value is provided, Amazon SageMaker will choose an optimal value for you
            "MaxConcurrentInvocationsPerInstance": endpoint_config.get(
                "max_concurrent_invocations", 4
            )
        },
    }
    return async_config


def create_config(endpoint_config, endpoint_type):
    """
    - Master function to create endpoint configuration.
    - This function will enable data_capture for real-time if data_capture_config is present
    - Data Capture is not supported for serverless and multi-model endpoints
    parameters:
        endpoint_config: endpoint configuration provided by user
        real_time: Boolean to indicate whether to use [real-time, multi-model] endpoints if True otherwise serverless endpoints

    Refer to this link for detailed description of an endpoint-configuration for all types of endpoint
     - https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html
    """
    if endpoint_type == "real-time-endpoint":
        if "data_capture_config" in endpoint_config:
            print(f"Data capture config Enabled")
            data_capture_config = get_data_capture_config(endpoint_config)
        else:
            data_capture_config = None
        product_config = get_real_time_config(endpoint_config)
        async_config = None
        response = create_config_util(
            endpoint_config, product_config, async_config, data_capture_config
        )
        print(f"Endpoint Configuration Created for {endpoint_type}")

    elif endpoint_type == "multi-model-endpoint":
        data_capture_config = None
        product_config = get_real_time_config(endpoint_config)
        async_config = None
        response = create_config_util(
            endpoint_config, product_config, async_config, data_capture_config
        )
        print(f"Endpoint Configuration Created for {endpoint_type}")

    elif endpoint_type == "serverless-endpoint":
        data_capture_config = None
        product_config = get_serverless_config(endpoint_config)
        async_config = None
        response = create_config_util(
            endpoint_config, product_config, async_config, data_capture_config
        )
        print(f"Endpoint Configuration Created for {endpoint_type}")

    elif endpoint_type == "async-endpoint":
        data_capture_config = None
        product_config = get_real_time_config(endpoint_config)
        async_config = get_async_config(endpoint_config)
        response = create_config_util(
            endpoint_config, product_config, async_config, data_capture_config
        )
        print(f"Endpoint Configuration Created for {endpoint_type}")
    else:
        supported_endpoints = [
            "real-time-endpoint",
            "multi-model-endpoint",
            "async-endpoint",
            "serverless-endpoint",
        ]
        supported_endpoints = ",".join(supported_endpoints)
        raise Exception(
            f"{endpoint_type} not supported. Currently Supporting : {supported_endpoints}"
        )
    print("{} Config Created".format(endpoint_config.get("config_name")))


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
    """
    Function to delete endpoint related resources
     - endpoint
     - endpoint configuration
     - model object
     parameters:
        endpoint_name - name of the endpoint
        config_name - name of the endpoint configuration
        model_name - name of the model object
        aws_region - AWS region in which the endpoint is hosted
    """
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
        help="Supported Endpoint Types : real-time-endpoint, multi-model-endpoint, serverless-endpoint, async-endpoint",
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
        "serverless-endpoint",
        "async-endpoint"
    ], f"Supported Endpoint Types are : real-time-endpoint,multi-model-endpoint, serverless-endpoint and async-endpoint"

    assert os.path.exists(config_path), f"{config_path} does not exist"

    endpoint_master_config = json.load(open(config_path))
    endpoint_config = endpoint_master_config.get(endpoint_type)

    if action == "create_endpoint":
        if endpoint_type in ["real-time-endpoint", "serverless-endpoint"]:
            ## This will create Model Object for real-time and serverless endpoints
            create_model(
                model_name=endpoint_config.get("model_name"),
                ecr_image=endpoint_config.get("container_uri"),
                model_uri=endpoint_config.get("model_uri"),
                role=endpoint_config.get("iam_role"),
                multi_model=False,
            )
        else:
            ## This will create Model Object for multi-model endpoints
            create_model(
                model_name=endpoint_config.get("model_name"),
                ecr_image=endpoint_config.get("container_uri"),
                model_uri=endpoint_config.get("model_uri"),
                role=endpoint_config.get("iam_role"),
                multi_model=True,
            )
        create_config(endpoint_config, endpoint_type=endpoint_type)
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
