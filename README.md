
# Sagemaker MlOps Pipeline

An end to end pipeline for labeling, training and deployment of an object detection model.

![image](https://user-images.githubusercontent.com/43469729/218812586-bc4acd09-c1c0-481a-b297-aca228b35828.png)
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

### Labelling Job

- Upload the Labelling Template [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/labelling/configs/instructions.template) on s3 
- Update the config : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/blob/main/labelling/configs/labelling_config.json). Config parameters reference : [Link](https://github.com/ayush9818/AWS-MLops-Pipeline/wiki/Sagemaker-Labelling-Jobs#parameters-description-) 
- To create a labelling job, run the following commands

  ```bash
    cd labelling
    python create_labelling_job.py --cfg configs/labelling_config.json
  ```
- Now worker will be assigned a labelling job, once the worker complete the task, an output.manifest file will be written on s3 which will be used for model training.

