import warnings
warnings.filterwarnings('ignore')


import os 
import argparse
import json 
from loguru import logger 
from ultralytics import YOLO


from prepare_dataset import prepare_dataset


# https://docs.ultralytics.com/cfg/ : Config parameters list for training
# pip install ultralytics


def parse_args():
    parser = argparse.ArgumentParser()
    # AWS
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/ml/checkpoints')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    args = parser.parse_args()
    return args


def train(data_config, model_config):
    model_config['data'] = data_config['yaml_file_path']
    model = YOLO(model_config.get('model','yolov8n.pt'))
    model.train(**model_config)

def main(aws_config, data_config, model_config):
    # Step 1 : Prepare dataset
    logger.info('Preparing dataset for Model Training')
    prepare_dataset(data_config=data_config, aws_config=aws_config)

    #Step 2 : Format model_config and train the model
    logger.info('Training Model')
    train(data_config, model_config)


if __name__ == '__main__':
    # Config file name should be config.json 
    args = parse_args()
    config_path = os.path.join(args.data_dir,'config.json')
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    config = json.load(open(config_path))
    data_config = config.get('data_config')
    model_config = config.get('model_config')
    aws_config = config.get('aws_config')

    main(aws_config, data_config, model_config)
