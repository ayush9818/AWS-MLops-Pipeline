import warnings
warnings.filterwarnings('ignore')


import os 
import argparse
import json 
from loguru import logger 
from ultralytics import YOLO
import shutil 
import xtarfile as tarfile 
import boto3

from prepare_dataset import prepare_dataset


# https://docs.ultralytics.com/cfg/ : Config parameters list for training
# pip install ultralytics

# TODO : Upload the final trained model weights on fixed location on s3
# TODO : Create a tar file for the final trained model weights which is to be used in inference endpoint

def parse_args():
    parser = argparse.ArgumentParser()
    # AWS
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    args = parser.parse_args()
    return args


def upload_file_on_s3(local_path, s3_path, bucket_name):
    my_bucket = boto3.resource('s3').Bucket(bucket_name)
    try:
        my_bucket.upload_file(local_path, s3_path)
        logger.info(f"Uploaded {local_path} to {s3_path}")
    except Exception as e:
        logger.error(f"Error uploading in {local_path} to {s3_path}")
        raise Exception(e)



def train(data_config, model_config):
    model_config['data'] = data_config['yaml_file_path']
    model = YOLO(model_config.get('model','yolov8n.pt'))
    model.train(**model_config)

def deploy_model(inference_config, deploy_config, model_dir, model_path, bucket_name):
    model_name = deploy_config.get('model_name')
    deploy_dir = deploy_config.get('deploy_dir')

    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir,exist_ok=True)
    
    assert os.path.exists(model_path), f"{model_path} does not exists"
    shutil.copy(model_path, model_dir)

    """
    Use of inference configuration is to demonstrate how we can use different pre-defined parameters
    for any custom inference pipeline. For example, the following paramters are available here:
         "iou": 0.7,
        "augment": true
    More parameters are available here: https://docs.ultralytics.com/cfg/
     """
    inference_config['weights'] = os.path.basename(model_path)
    with open(os.path.join(model_dir,'model.json'),'w') as f:
        json.dump(inference_config, f)

    save_dir = os.path.dirname(model_path)
    tar_file_path = os.path.join(save_dir,f'{model_name}.tar.gz')
    with tarfile.open(tar_file_path, 'w:gz') as tar:
        tar.add(model_dir,arcname='.')

    upload_file_on_s3(
        local_path=tar_file_path,
        s3_path=os.path.join(deploy_dir,f"{model_name}.tar.gz"),
        bucket_name=bucket_name)



def main(aws_config, data_config, model_config, inference_config, deploy_config):
    # Step 1 : Prepare dataset
    logger.info('Preparing dataset for Model Training')
    prepare_dataset(data_config=data_config, aws_config=aws_config)

    #Step 2 : Format model_config and train the model
    logger.info('Training Model')
    train(data_config, model_config)

    #Step3 : Upload Model Artifacts on s3
    logger.info(f"Deploying model artifacts on s3")
    model_path = os.path.abspath('runs/detect/train/weights/best.pt')
    deploy_model(
        inference_config=inference_config,
        deploy_config=deploy_config,
        model_dir=data_config.get('model_dir'),
        model_path=model_path,
        bucket_name=aws_config.get('bucket_name')
    )


if __name__ == '__main__':
    # Config file name should be config.json 
    args = parse_args()
    config_path = os.path.join(args.data_dir,'config.json')
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    config = json.load(open(config_path))
    data_config = config.get('data_config')
    model_config = config.get('model_config')
    aws_config = config.get('aws_config')
    deploy_config = config.get('deploy_config')
    inference_config = config.get('inference_config')

    main(aws_config, data_config, model_config, inference_config, deploy_config)

