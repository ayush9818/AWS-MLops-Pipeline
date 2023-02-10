import os 
import argparse
import json 
import pandas as pd
from random import shuffle


def format_annotation(img_h, img_w, x_top, y_top, box_w, box_h, class_id):
    center_x = round((x_top + (box_w / 2)) / img_w,4)
    center_y = round((y_top + (box_h / 2)) / img_h,4)
    w = round(box_w / img_w,4)
    h = round(box_h / img_h,4)
    anno_dict = {
        'label' : class_id,
        'center_x' : center_x,
        'center_y' : center_y,
        'w' : w,
        'h' : h
    }
    return anno_dict
                

def get_annotations(anno_info, label_info):
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
        
def prepare_data_df(dataset_path):
    with open(dataset_path,'r') as f:
        dataset = f.read().split('\n')
    image_path_list = []
    image_name_list = []
    anno_list = []
    is_background_list = []

    class_mapping = {} # TO BE IMPLEMENTED
    for image_data in dataset[:-1]:
        image_data = json.loads(image_data)
        image_path = image_data.get('source-ref')
        anno_info = image_data.get('category')
        label_info = image_data.get('category-metadata').get('class-map')
        image_anno_list = get_annotations(anno_info, label_info)
        
        image_path_list.append(image_path)
        image_name_list.append(os.path.basename(image_path))
        anno_list.append(image_anno_list)
        if len(image_anno_list) == 0:
            is_background_list.append(True)
        else:
            is_background_list.append(False)
    data_df = pd.DataFrame({
        "image_name" : image_name_list,
        "image_path" : image_path_list,
        "annotations" : anno_list,
        "is_background" : is_background_list
        }
    )
    return data_df, class_mapping


def split_dataset(data_df, split_dict):
    if split_dict is None:
        split_dict = {'train' : 0.8, 'valid' : 0.2}

    total_len = data_df.shape[0]
    train_count = int(split_dict.get('train') * total_len)
    valid_count = total_len - train_count
    
    train_type_list = ['train']*train_count + ['valid']*valid_count
    shuffle(train_type_list)
    data_df['train_type'] = train_type_list
    return data_df

def prepare_yolo_annotations(data_df, class_mapping, train_path, valid_path):
    pass

def prepare_dataset():
    pass



if __name__ == '__main__':
    pass