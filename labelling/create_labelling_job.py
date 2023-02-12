import os
import sys
import sagemaker
import pandas as pd
import boto3
import numpy as np
import json
import warnings
import argparse 
from loguru import logger
warnings.filterwarnings('ignore')


def generate_manifest_file(bucket_name, dataset_path, manifest_upload_dir, manifest_file_name):
    """
    Generates a manifest file containing the location of images used in the labelling job
    params:
        bucket_name : s3 bucket name 
        dataset_path : relative s3 path of the image dataset 
        manifest_upload_dir : s3 directory to upload the manifest file
        manifest_file_name : name of the manifest file 
    """
    image_extensions = ['png', 'jpg', 'jpeg']
    local_manifest_file_path = os.path.join(os.getcwd(), manifest_file_name)
    s3 = boto3.resource('s3')
    dataset_bucket = s3.Bucket(bucket_name)
    with open(local_manifest_file_path,'w') as outfile:
        for object_summary in dataset_bucket.objects.filter(Prefix=dataset_path):
            object_key = object_summary.key
            file_extension  = object_key.split('.')[-1]
            if file_extension in image_extensions:
                file_name  = object_key.split('/')[-1]
                file_path = os.path.join(f"s3://{bucket_name}",dataset_path,file_name)
                data_dict = {"source-ref": file_path}
                outfile.write(json.dumps(data_dict) + "\n") 
    logger.info(f"Manifest File Creation Done. Uploading Manifest file to : {manifest_upload_dir}")
    try:
        dataset_bucket.upload_file(local_manifest_file_path, os.path.join(manifest_upload_dir,manifest_file_name))
    except Exception as e:
        raise Exception(f"Failed to Upload {local_manifest_file_path} to {manifest_upload_dir}\nError : {e}")
    logger.info(f"Upload Successful")
    os.remove(local_manifest_file_path)


def generate_label_file(bucket_name, label_list, label_file_name, label_file_upload_dir):
    """
    Generate a json file containing information of labels to be annotated and upload on S3
    params:
        bucket_name : S3 Bucket Name
        label_list : list of labels to be annotated 
        label_file_name : name of label file 
        label_file_upload_dir : s3 directory to upload label file
    """
    label_dict = {"labels": [{"label": label} for label in label_list]}
    local_label_file_path = os.path.join(os.getcwd(), label_file_name)
    with open(local_label_file_path,'w') as f:
        json.dump(label_dict,f)
        
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(bucket_name)
    logger.info(f"Uploading Label File {label_file_name} to {label_file_upload_dir}")
    try:
        s3_bucket.upload_file(local_label_file_path, os.path.join(label_file_upload_dir,label_file_name))
    except Exception as e:
        raise Exception(f"Failed to upload {local_label_file_path} to {label_file_upload_dir}")
    logger.info("Uploaded Successfully")
    os.remove(local_label_file_path)

def create_human_task_config(acs_arn,
                             pre_human_arn,
                             MaxConcurrentTaskCount,
                             NumberOfHumanWorkersPerDataObject,
                             TaskAvailabilityLifetimeInSeconds,
                             TaskTimeLimitInSeconds,
                             TaskDescription,
                             TaskKeywords,
                             TaskTitle,
                             template_file_uri,
                             work_team_arn
                            ):
    """
    The function will create a config defining certain rules and parameters for human labellers
    params:
        acs_arn : ACS arn for bounding box job in ap-southeast-1
        pre_human_arn : Pre-Human arn for bounding box job in ap-southeast-1
        MaxConcurrentTaskCount : Images sent at a time to the workteam
        NumberOfHumanWorkersPerDataObject : Workers to label each image
        TaskAvailabilityLifetimeInSeconds : Time to complete all pending tasks
        TaskTimeLimitInSeconds :  Time to complete each image
        TaskDescription : Brief description of the task 
        TaskKeywords : Keywords related to Task
        TaskTitle : Title of the task,
        template_file_uri : Template of the file containing description and rules for the job
        work_team_arn : 
    returns:
        human_task_config
    """
    human_task_config = {
        "AnnotationConsolidationConfig": {
            "AnnotationConsolidationLambdaArn": acs_arn,
        },
        "PreHumanTaskLambdaArn": pre_human_arn,
        "MaxConcurrentTaskCount": MaxConcurrentTaskCount, 
        "NumberOfHumanWorkersPerDataObject": NumberOfHumanWorkersPerDataObject,
        "TaskAvailabilityLifetimeInSeconds": TaskAvailabilityLifetimeInSeconds, 
        "TaskDescription": TaskDescription,
        "TaskKeywords": TaskKeywords,
        "TaskTimeLimitInSeconds": TaskTimeLimitInSeconds,  
        "TaskTitle": TaskTitle,
        "UiConfig": {
            "UiTemplateS3Uri": template_file_uri,
        },
        "WorkteamArn" : work_team_arn
    }
    return human_task_config

def create_ground_truth_request(manifest_file_uri,
                                output_path_uri,
                                human_task_config,
                                job_name,
                                iam_role,
                                label_file_uri
                               ):
    """
    Generates a ground truth request dictionary to create a labelling job
    """
    ground_truth_request = {
        "InputConfig": {
            "DataSource": {
                "S3DataSource": {
                    "ManifestS3Uri": manifest_file_uri,
                }
            },
            "DataAttributes": {
                "ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation", "FreeOfAdultContent"]
            },
        },
        "OutputConfig": {
            "S3OutputPath": output_path_uri,
        },
        "HumanTaskConfig": human_task_config,
        "LabelingJobName": job_name,
        "RoleArn": iam_role,
        "LabelAttributeName": "category",
        "LabelCategoryConfigS3Uri": label_file_uri,
    }
    return ground_truth_request

def main(aws_config, label_config):
    # Step 1 : Generate and upload a manifest file of the input dataset in s3 bucket
    generate_manifest_file(
        bucket_name=aws_config.get('bucket_name'),
        dataset_path=label_config.get('dataset_path'),
        manifest_upload_dir=label_config.get('manifest_upload_dir'),
        manifest_file_name=label_config.get('manifest_file_name')
    )

    # Step 2 : Generate a json file containing class_names information
    generate_label_file(
        bucket_name=aws_config.get('bucket_name'),
        label_list=label_config.get('label_list'),
        label_file_name=label_config.get('label_file_name'),
        label_file_upload_dir=label_config.get('label_file_upload_dir')
    )

    # Step 3 : Create a human task config containing information about labelling for a worker 
    human_task_config = create_human_task_config(
        acs_arn=aws_config.get('acs_arn'),
        pre_human_arn=aws_config.get('pre_human_arn'),
        MaxConcurrentTaskCount=label_config.get('max_concurrent_task_count'),
        NumberOfHumanWorkersPerDataObject=label_config.get('number_of_human_workers'),
        TaskAvailabilityLifetimeInSeconds=label_config.get('task_availability_lifetime'),
        TaskTimeLimitInSecond=label_config.get('task_time_limit'),
        TaskDescription=label_config.get('task_description'),
        TaskKeywords=label_config.get('task_keywords'),
        TaskTitle=label_config.get('task_title'),
        template_file_uri=label_config.get('template_file_uri'),
        work_team_arn=aws_config.get('private_work_team_arn'),
    )

    # Step 4 : Create a request body to create a labelling job 
    bucket_name = aws_config.get('bucket_name')
    manifest_file_uri = os.path.join(f"s3://{bucket_name}",label_config['manifest_file_upload_dir'],label_config['manifest_file_name'])
    output_path_uri = os.path.join(f"s3://{bucket_name}",label_config['output_dir'])
    label_file_uri = os.path.join(f"s3://{bucket_name}",label_config['label_file_upload_dir'],label_config['label_file_name'])
    ground_truth_request = create_ground_truth_request(
                                manifest_file_uri=manifest_file_uri,
                                output_path_uri=output_path_uri,
                                human_task_config=human_task_config,
                                job_name=label_config.get('job_name'),
                                iam_role=aws_config.get('iam_role'),
                                label_file_uri=label_file_uri
                            )

    logger.info(f"Ground Truth Request : {ground_truth_request}")

    # Step 5 : Create a Labeling Job
    logger.info(f"Creating Labeling Job")
    sagemaker_client = boto3.client("sagemaker")
    job_info = sagemaker_client.create_labeling_job(**ground_truth_request)
    logger.info(f"Created Labeling Job. Job information : {job_info}")

    job_status = sagemaker_client.describe_labeling_job(LabelingJobName=label_config.get('job_name'))["LabelingJobStatus"]
    print(job_status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str,help='path of configuration file')

    args=parser.parse_args()

    config_path = args.cfg
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    config = json.load(open(config_path))
    aws_config = config.get('aws_config')
    label_config = config.get('label_config')

    logger.info(f"aws_config: {aws_config}\nlabel_config: {label_config}")
    main(aws_config, label_config)