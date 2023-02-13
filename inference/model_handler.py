import os
import json
import time
from time import sleep
import torch 
from loguru import logger
from ultralytics import YOLO
import boto3

curr_dir = os.path.abspath(os.path.dirname(__file__))

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """
    def __init__(self):
        self.initialized = False
        self.model = None

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
        if os.path.exists(s3_path):
            return s3_path
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket_name)
        try:
            my_bucket.download_file(s3_path, local_path)
            logger.info(f"Image Downloaded to {local_path}") 
        except Exception as e:
            logger.error(f"Error downloading {s3_path} to {local_path}")
            raise Exception(e)
        return local_path

    def format_output(self, model_output):
        model_output = model_output[0].boxes 
        bounding_boxes = model_output.xywhn.cpu().tolist()
        conf_scores = model_output.conf.cpu().tolist()
        class_ids = model_output.cls.cpu().tolist()
        result_list = []
        for box, conf, class_id in zip(bounding_boxes, conf_scores, class_ids):
            tmp_dict = {
                "coordinates" : {
                    "x" : box[0], "y" : box[1], "w" : box[2], "h" : box[3]
                },
                "class_id" : int(class_id),
                "confidence" : conf 
            }
            result_list.append(tmp_dict)
        return result_list
    
    def handle(self, data, context):
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        logger.info(f"Running Inference")
        image_dict = json.loads(data[0].get('body'))
        image_path = image_dict.get('image_path')
        bucket_name = image_dict.get('bucket_name')
        conf = image_dict.get('conf')

        local_path = os.path.join(curr_dir, os.path.basename(image_path))
        local_path = self.download_image(s3_path=image_path, local_path=local_path, bucket_name=bucket_name)

        self.model_dict['params']['conf'] = conf
        self.model_dict['params']['device'] = device
        model_output = self.model.predict(image_path, **self.model_dict.get('params'))
        model_output = self.format_output(model_output)
        image_dict['results'] = model_output

        os.remove(local_path)
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

if __name__ == '__main__':
    import json
    payload = json.load(open('sample_files/payload.json'))
    print(handle(payload, context={}))