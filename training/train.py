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

def train(data_config, model_config):
    model_config['data'] = data_config['yaml_file_path']
    model = YOLO(model_config.get('model','yolov8n.pt'))
    model.train(**model_config)

def main(aws_config, data_config, model_config):
    # Step 1 : Prepare dataset
    prepare_dataset(data_config=data_config, aws_config=aws_config)

    #Step 2 : Format model_config and train the model
    train(data_config, model_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Configuration file for training')

    args = parser.parse_args()
    config_path = args.cfg 
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    config = json.load(open(config_path))
    data_config = config.get('data_config')
    model_config = config.get('model_config')
    aws_config = config.get('aws_config')

    main(aws_config, data_config, model_config)
