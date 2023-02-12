import os 
import argparse
import json 
import pandas as pd
from random import shuffle
import boto3
import yaml
from loguru import logger

import warnings
warnings.filterwarnings('ignore')

"""
TO BE DONE 
    Parallel Image Download
    Reindex class_mapping from scratch
"""

def format_annotation(img_h, img_w, x_top, y_top, box_w, box_h, class_id):
    """
    Convert the Sagemaker GroundTruth Annotations to Yolo annotations 
    Sagemaker GroundTruth Annotations : [x_top, y_top, box_w, box_h] [Abosolute]
    Yolo Annotations : [x_center, y_center, w, h] [Normalized]
    params:
        img_h, img_w : image_height, image_width
        x_top : X coordinate of top left corner of the bounding box
        y_top : Y coordinate of top left corner of the bounding box
        box_w : Width of the bounding box
        box_h : Height of the bounding box
        class_id : class_number of the label class
    returns:
        anno_dict : Yolo formatted annotations dictionary of a boudning box
    """
    center_x = round((x_top + (box_w / 2)) / img_w,4)
    center_y = round((y_top + (box_h / 2)) / img_h,4)
    w = round(box_w / img_w,4)
    h = round(box_h / img_h,4)
    anno_dict = {
        'label' : int(class_id),
        'center_x' : center_x,
        'center_y' : center_y,
        'w' : w,
        'h' : h
    }
    return anno_dict
                
def get_annotations(anno_info):
    """
    Extracts the annotations from sagemaker manifest file format and returns yolo formatted annotations
    params:
        anno_info : sagemaker groundtruth manifest dictionary of an object containing the annotations
    returns:
        temp_anno_list : list of yolo fomatted annotations for an image file
    """
    img_h = anno_info.get('image_size')[0].get('height')
    img_w = anno_info.get('image_size')[0].get('width')
    temp_anno_list = []
    for anno in anno_info.get('annotations'):
        class_id = anno.get('class_id')
        x_top = anno.get('left')
        y_top = anno.get('top')
        box_w = anno.get('width')
        box_h = anno.get('height')
        anno_dict = format_annotation(img_h, img_w, x_top, y_top, box_w, box_h, class_id)
        temp_anno_list.append(anno_dict)
    return temp_anno_list
        
def get_class_mapping(label_info, class_mapping):
    """
    Prepare a class mapping for the manifest file info
    params:
        label_info : manifest file label map
        class_mapping : old class mapping
    returns:
        class_mapping : new class mapping
    """
    for class_id, class_name in label_info.items():
        if int(class_id) not in class_mapping:
            class_mapping[int(class_id)] = class_name
    return class_mapping

def prepare_data_df(dataset_path):
    """
    Prepare a dataframe containing annotations of all images and image info for the input dataset manifest file
    params : 
        dataset_path : path of the sagemaker ground truth manifest file
    returns:
        data_df : dataframe containing image and annotation information
        class_mapping : dictionary mapping of class_id to class_name
    """
    with open(dataset_path,'r') as f:
        dataset = f.read().split('\n')
    image_path_list = []
    image_name_list = []
    anno_list = []
    is_background_list = []

    class_mapping = {} 
    for image_data in dataset[:-1]:
        image_data = json.loads(image_data)
        image_path = image_data.get('source-ref')
        anno_info = image_data.get('category')
        label_info = image_data.get('category-metadata').get('class-map')
        image_anno_list = get_annotations(anno_info)
        class_mapping = get_class_mapping(label_info, class_mapping)
        image_path_list.append(image_path)
        image_name_list.append(os.path.basename(image_path))
        anno_list.append(image_anno_list)
        if len(image_anno_list) == 0:
            is_background_list.append(True)
        else:
            is_background_list.append(False)
    data_df = pd.DataFrame({
        "image_name" : image_name_list,
        "s3_path" : image_path_list,
        "annotations" : anno_list,
        "is_background" : is_background_list
        }
    )
    return data_df, class_mapping

def split_dataset(data_df, split_dict):
    """
    Split the input dataset into train and validations sets
    params:
        data_df : input dataframe of the dataset
        split_dict : dictionary containing split percentage of train and validation sets
    returns:
        data_df : output dataframe after splitting into train and validations sets
    """
    if split_dict is None:
        split_dict = {'train' : 0.8, 'valid' : 0.2}

    total_len = data_df.shape[0]
    train_count = int(split_dict.get('train') * total_len)
    valid_count = total_len - train_count
    
    train_type_list = ['train']*train_count + ['valid']*valid_count
    shuffle(train_type_list)
    data_df['train_type'] = train_type_list
    return data_df

def prepare_label_files(df, label_save_path):
    """
    Prepare txt files for annotations. No .txt files are saved for background images with flag is_background=TRUE
    params:
        df : input dataframe containing image info and annotations
        label_save_path : save_dir to save the txt files
    """
    for idx,row in df.iterrows():
        if row['is_background']:
            continue
        image_path = row['s3_path']
        image_name = os.path.basename(image_path)
        image_extension = image_name.split('.')[-1]
        label_file_name = image_name.replace(f".{image_extension}",".txt")
        with open(os.path.join(label_save_path,label_file_name),'w') as f:
            for anno in row['annotations']:
                f.writelines(f"{anno['label']} {anno['center_x']} {anno['center_y']} {anno['w']} {anno['h']}\n")
    
def prepare_yaml_file(train_images_path, valid_images_path, class_mapping, save_path):
    """Prepare the yaml file for training"""
    data_summary = {
        "train" : train_images_path,
        "val" : valid_images_path,
        "names" : class_mapping
    }
    with open(save_path,'w') as f:
        yaml.dump(data_summary, f)

def prepare_yolo_annotations(data_df, class_mapping, save_dir, yaml_save_path):
    logger.info(f"Dataset Save Directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    train_image_path = os.path.abspath(os.path.join(save_dir,'images/train'))
    valid_image_path = os.path.abspath(os.path.join(save_dir,'images/val'))
    train_labels_path = os.path.abspath(os.path.join(save_dir,'labels/train'))
    valid_labels_path = os.path.abspath(os.path.join(save_dir,'labels/val'))
    logger.info(f"Train Images Directory: {train_image_path}\nValid Images Directory: {valid_image_path}")
    logger.info(f"Train labels Directory: {train_labels_path}\nValid Labels Directory : {valid_labels_path}")
    os.makedirs(train_image_path,exist_ok=True)
    os.makedirs(valid_image_path,exist_ok=True)
    os.makedirs(train_labels_path,exist_ok=True)
    os.makedirs(valid_labels_path,exist_ok=True)

    train_df = data_df[data_df.train_type=='train']
    valid_df = data_df[data_df.train_type=='valid']

    logger.info(f"Preparing txt files for Training Images")
    prepare_label_files(df=train_df, label_save_path=train_labels_path)

    logger.info(f"Preparing txt files for Validation Images")
    prepare_label_files(df=valid_df, label_save_path=valid_labels_path)

    # Appending local paths for images in train and validation dataframes
    train_df['image_path'] = train_df['s3_path'].apply(lambda x: os.path.join(train_image_path,os.path.basename(x)))
    valid_df['image_path'] = valid_df['s3_path'].apply(lambda x: os.path.join(valid_image_path,os.path.basename(x)))

    data_df = pd.concat([train_df,valid_df])
    #data_df.to_json(os.path.join(save_dir,'data_df.json'))

    logger.info(f"Preparing Data Yaml File for training")
    prepare_yaml_file(train_image_path, valid_image_path, class_mapping, yaml_save_path)
    return data_df 

def download_dataset(data_df, bucket_name):
    logger.info(f"Downloading dataset")
    bucket_prefix = f"s3://{bucket_name}/"
    data_df['s3_path'] = data_df['s3_path'].apply(lambda x : x.replace(bucket_prefix,''))
    total_images = data_df.shape[0]
    done = 0
    files_to_remove = []
    for idx,row in data_df.iterrows():
        s3_path = row['s3_path']
        local_path = row['image_path']
        response = download_files_from_s3(s3_path, local_path, bucket_name)
        if response is None:
            files_to_remove.append(local_path)
            continue
        if done % 50 == 0:
            logger.info(f"{done}/{total_images} finished. {len(files_to_remove)}/{total_images} Failed")
    data_df = data_df[~data_df.image_path.isin(files_to_remove)]
    return data_df

def download_files_from_s3(s3_path, local_path, bucket_name):
    if os.path.exists(s3_path):
        return s3_path
    my_bucket = boto3.resource('s3').Bucket(bucket_name)
    try:
        base_dir = os.path.dirname(local_path)
        os.makedirs(base_dir,exist_ok=True)
        my_bucket.download(s3_path, local_path)
    except Exception as e:
        logger.error(f"Error downloading {s3_path}. Error: {e}")
        return None
    return local_path

def prepare_dataset(data_config, aws_config):
    # Step 1 : Prepare the dataset df containing annotations and image_info
    save_dir = data_config.get('dataset_save_dir')
    dataset_s3_path = data_config.get('input_dataset_path')
    dataset_name = os.path.basename(dataset_s3_path)
    dataset_local_path = os.path.join(save_dir, dataset_name)
    dataset_local_path = download_files_from_s3(dataset_s3_path, dataset_local_path, aws_config.get('bucket_name'))
    data_df, class_mapping = prepare_data_df(dataset_path=dataset_local_path)

    # Step 2 : Split the dataframe into train and validation sets 
    data_df = split_dataset(data_df=data_df, split_dict=data_config.get('split_dict',None))

    # Step 3 : Generate txt annotations file and data yaml file for model training 
    data_df = prepare_yolo_annotations(
        data_df = data_df, 
        class_mapping=class_mapping,
        save_dir=data_config.get('dataset_save_dir'),
        yaml_save_path=data_config.get('yaml_file_path')
    )

    # Step 4 : Download the images into local from s3 
    data_df = download_dataset(data_df=data_df, bucket_name=aws_config.get('bucket_name'))
    data_df.to_json(os.path.join(save_dir,'data_df.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='dataset configuration file path')

    args = parser.parse_args()
    config_path = args.cfg
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    config = json.load(open(config_path))
    data_config = config.get('data_config')
    aws_config = config.get('aws_config')
    logger.info(f"AWS configuration : {aws_config}\n Data Conguration : {data_config}")

    logger.info("Preparaing Dataset for Yolo Training")
    prepare_dataset(data_config, aws_config)