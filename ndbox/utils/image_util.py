import os
import time
from copy import deepcopy
from os import path

from ndbox.dataset import build_dataset
from ndbox.processor import build_processor
from ndbox.utils import dict2yaml


def save_image(save_path, nwb_data, dataset_opt, processor_opt=None):
    config_path = path.join(save_path, 'config.yml')
    save_dict = {'dataset': dataset_opt}
    if processor_opt is not None:
        save_dict['processor'] = processor_opt

    with open(config_path, 'w', encoding='utf-8') as f:
        rs = dict2yaml(save_dict)
        f.write(rs)

    nwb_data.save_image(save_path)


def load_image(temp_path, dataset_opt, processor_opt=None):
    image_path = find_image_path(temp_path, dataset_opt, processor_opt)
    if image_path is not None:
        opt = deepcopy(dataset_opt)
        opt['image_path'] = image_path
        nwb_data = build_dataset(opt)
    else:
        opt = deepcopy(dataset_opt)
        nwb_data = build_dataset(opt)
        prefix = 'd_'
        if processor_opt is not None:
            prefix = 'p_'
            build_processor(nwb_data, processor_opt)
        save_path = path.join(temp_path, f"{prefix}_image_{int(time.time())}")
        os.makedirs(save_path)
        save_image(save_path, nwb_data, dataset_opt, processor_opt)
    return nwb_data


def restore_image(temp_path, nwb_data, dataset_opt, processor_opt=None):
    image_path = find_image_path(temp_path, dataset_opt, processor_opt)
    if image_path is not None:
        nwb_data.restore_image(image_path)
    else:
        prefix = 'd_'
        if processor_opt is not None:
            prefix = 'p_'
            build_processor(nwb_data, processor_opt)
        save_path = path.join(temp_path, f"{prefix}_image_{int(time.time())}")
        os.makedirs(save_path)
        save_image(save_path, nwb_data, dataset_opt, processor_opt)


def find_image_path(temp_path, dataset_opt, processor_opt=None):
    load_dict = {'dataset': dataset_opt}
    if processor_opt is not None:
        load_dict['processor'] = processor_opt
    target_rs = dict2yaml(load_dict)

    images_dir = [d for d in os.listdir(temp_path) if path.isdir(path.join(temp_path, d))]
    for image_dir in images_dir:
        image_path = path.join(temp_path, image_dir)
        config_path = path.join(image_path, 'config.yml')
        if path.exists(config_path) and path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                rs = str(f.read())
                if target_rs == rs:
                    return image_path
    return None
