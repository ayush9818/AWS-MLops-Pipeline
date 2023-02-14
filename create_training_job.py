import time
import json
import os
import sagemaker as sage
from sagemaker.estimator import Estimator
from loguru import logger 
import argparse


class AWS_Job_Scheduler:
    """Schedule Sagemaker Training Jobs"""

    def __init__(self, job_config):
        """
        args:
            ts_id : training_session_id
            model_id : model_id of the new model
            data_dir : local data dir where training meta data is stored
        """
        self.job_config = job_config
        self.job_name = f'{job_config.get("base_job_name")}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}'
        self.job_dir = f"s3://{job_config.get('bucket_name')}/{os.path.join(job_config.get('job_dir'),self.job_name)}"
        self.local_data_dir = job_config.get('data_dir')
        self.data_prefix = f"{job_config.get('job_dir')}/{self.job_name}"
        logger.info(f"Job Configuration : {self.job_config}")
        logger.info(f"Job Name : {self.job_name}, Job Directory : {self.job_dir}")


    def schedule_job(self):
        sess = sage.Session(default_bucket=self.job_config.get('bucket_name'))
        data_location = sess.upload_data(self.local_data_dir, key_prefix=self.data_prefix)
        estimator = Estimator(
            role=self.job_config.get('aws_role'),
            image_uri=self.job_config.get('ecr_image_uri'),
            output_path=self.job_dir,
            checkpoint_s3_uri='{}/{}/'.format(self.job_dir, 'checkpoints'),
            instance_count=self.job_config.get('instance_count'),
            instance_type=self.job_config.get('instance_type')
        )
        try:
            estimator.fit(data_location, job_name=self.job_name, wait=True, logs='All')
        except Exception as e:
            print(e)
            return False

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help='job configuration file path')
    args = parser.parse_args()

    config_path = args.cfg 
    assert os.path.exists(config_path), f"Job configuration file path {config_path} does not exist"

    job_config = json.load(open(config_path))
    job_scheduler = AWS_Job_Scheduler(job_config)
    job_config = job_scheduler.schedule_job()
