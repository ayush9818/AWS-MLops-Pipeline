
# Sagemaker MLOps Pipeline

An end to end pipeline for labeling, training and deployment of an object detection model.

![Image](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/configs/flow_chart.png)

## Installation

Cloning the repository and setting up virtual environment

```bash
  git clone https://github.com/ayush9818/AWS-MLops-Pipeline
  cd AWS-MLops-Pipeline
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```
    
## Features

- Labelling Objects using Sagemaker Ground Truth
- Custom Training Jobs on Sagemaker
- Deployment of custom models using Real Time Endpoint on Sagemaker
- Inference using Real Time Endpoint


## Run Locally
<details>
  <summary>Click me</summary>
  
  ### Labelling Job

  - Upload the Labelling Template [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/labelling/configs/instructions.template) on s3 

  - Update the config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/labelling/configs/labelling_config.json). Config parameters reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Labelling-Jobs#parameters-description-) 
  - To create a labelling job, run the following commands

    ```bash
      cd labelling
      python create_labelling_job.py --cfg configs/labelling_config.json
    ```
  - Now worker will be assigned a labelling job, once the worker complete the task, an output.manifest file will be written on s3 which will be used for model training.
  ----

  ### Training Job 

  #### Building Training Container Image in local
  - Use GPU base image to enable GPU support.
  ```bash
    cd training
    docker build -f Dockerfiles/AwsCPUDockerfile -t <image_name> .
  ```
  #### Pushing the image to ECR
  - Need to add ECR FullAccessRole Policy in Sagemaker IAM Role
  - Login into ECR repo using commands on ECR UI 
  - To tag and push the image, run the following commands 
  ```bash
    docker tag <image_name> <ecr_repo_uri>:<tag_name>
    docker push <ecr_repo_uri>:<tag_name>
  ```

  #### Creating a Training Job on  Sagemaker

  - Prepare a Job Config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/configs/job_config.json). Parameters Reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Training-Jobs#parameters-description-for-job-config-)
  - Prepare a Train Config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/data/config.json). Parameters Reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Training-Jobs#parameters-description-for-train-config-)
  - To create a training job, run the following commands
  ```bash
    cd AWS-MLops-Pipeline
    python create_training_job.py --cfg configs/job_config.json
  ```
  ----

  ### Endpoint Deployment
  #### Building Inference Container Image in local
  - Use GPU base image to enable GPU support.
  ```bash
    cd inference
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
    docker build -f Dockerfiles/CpuDockerfile -t <image_name> .
  ```
  #### Pushing the image to ECR
  - Need to add ECR FullAccessRole Policy in Sagemaker IAM Role
  - Login into ECR repo using commands on ECR UI 
  - To tag and push the image, run the following commands 
  ```bash
    docker tag <image_name> <ecr_repo_uri>:<tag_name>
    docker push <ecr_repo_uri>:<tag_name>
  ```

  #### Creating a Real Time Endpoint
  - Prepare an endpoint config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/configs/endpoint_config.json). Parameters Reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Inference#parameters-description-of-endpoint_config)
  - To create an inference endpoint, run the following commands
  ```bash
    cd AWS-MLops-Pipeline
    python inference_resources.py --cfg configs/endpoint_config.json --action create_endpoint --endpoint-type real-time-endpoint/multi-model-endpoint/serverless-endpoint
  ```

  #### Deleting the Endpoint Resources 
  ```bash
    python inference_resources.py --cfg configs/endpoint_config.json --action delete_endpoint --endpoint-type real-time-endpoint/multi-model-endpoint/serverless-endpoint
  ```
  ----
  ### Model Inference 
  - Create a inference config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/configs/inference_config.json). Parameters Reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Inference#parameters-description-of-inference_config)
  ```bash
    cd AWS-MLops-Pipeline
    ## This script enables Data Capture for Real Time Endpoints and uses Sagemaker Predictor class
    python invoke_endpoint_v2.py --cfg configs/inference_config.json --endpoint-type real-time-endpoint/multi-model-endpoint/serverless-endpoint

    ## This script is based on invoking endpoints using Boto3 and do not enable Data Capture for Real Time Endpoints
    python invoke_endpoint.py --cfg configs/inference_config.json --endpoint-type real-time-endpoint/multi-model-endpoint/serverless-endpoint
  ```
</details>

## Authors

- [Ayush Agarwal](https://www.github.com/ayush9818)

