import os
import sys
import json
import time
from time import sleep
import torch 
from loguru import logger
from ultralytics import YOLO

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """
    def __init__(self):
        self.initialized = False
        self.defect_detector = None

    def initialize(self, context):
        self.initialized = True

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        model_path = os.path.join(model_dir, 'model.json')
        logger.info(f"ModelDir={model_dir} ModelPath={model_path}")

        self.model_dict=json.load(open(model_path))
        self.model_dict['weights'] = os.path.join(model_dir, self.model_dict.get('weights'))
        self.model = YOLO(self.model_dict.get('weights'))
        logger.info(f"Model Loaded")

    def download_image(self, s3_path, local_path, bucket_name):
        pass

    def format_output(self, model_output):
        result = None
        return result
    
    def handle(self, data, context):
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        logger.info(f"Running Inference")
        image_dict = json.loads(data[0].get('body'))
        # Download file 
        # download_image()
        image_path = None
        model_output = self.model.predict(image_path, **self.model_dict.get('params'))
        model_output = self.format_output(model_output)

        image_dict['results'] = model_output
        return image_dict
    
_service = ModelHandler()

def handle(data, context):
    logger.info(f"Request Received\nData={data}\nContext={context}")
    is_initialized = 'no'
    if not _service.initialized:
        _service.initialize(context)
        is_initialized = 'yes'

    if data is None:
        return None
    
    out = _service.handle(data, context)
    out['is_initalized'] = is_initialized
    return [out]