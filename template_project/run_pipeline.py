import datetime
import os
import random
import argparse
import time
from os import path
from copy import deepcopy

from ndbox.dataset import build_dataset
from ndbox.processor import run_processor
from ndbox.model import build_model
from ndbox.utils import yaml_load, opt2str, set_random_seed, get_root_logger

try:
    import user_define_modules
except ImportError:
    pass


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    default_config = path.join(root_path, 'config.yml')
    parser.add_argument('-opt', type=str, default=default_config, help='Config path.')
    args = parser.parse_args()
    opt = yaml_load(args.opt)

    opt['path'] = {}
    name = opt.get('name', 'result')
    result_path = path.join(root_path, f'{name}_{int(time.time())}')
    log_path = path.join(result_path, 'log.txt')

    opt['path']['root'] = root_path
    opt['path']['result'] = result_path
    opt['path']['log'] = log_path
    os.makedirs(result_path)

    seed = opt.get('random_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['random_seed'] = seed
    set_random_seed(seed)

    return opt, args


def run_pipeline(root_path):
    opt, args = parse_options(root_path)
    logger = get_root_logger(log_file=opt['path']['log'])
    logger.info(opt2str(opt))

    dataset = {}
    for dataset_name, dataset_opt in opt.get('dataset', {}).items():
        dataset[dataset_name] = build_dataset(opt=dataset_opt)

    result_path = opt['path']['result']
    for exp_name, exp_opt in opt.get('experiment', {}).items():
        exp_path = path.join(result_path, exp_name)
        os.makedirs(exp_path)
        logger.info(f'Start experiment {exp_name}.')

        processor_opt = exp_opt.get('processor', {})
        model_opt = exp_opt.get('model', {})
        train_opt = exp_opt.get('train', {})
        test_opt = exp_opt.get('test', {})
        metric_list = exp_opt.get('metrics', {})
        p_dataset = processor_pipeline(dataset, processor_opt)

        # train model
        model_load_path = model_opt.get('path')
        model_list = []
        if model_load_path is not None:
            model = build_model(model_opt)
            model.load(model_load_path)
            model_list.append(model)
        else:
            train_data_name = train_opt.get('dataset')
            if train_data_name is not None:
                train_data = p_dataset.get(train_data_name)
                if train_data is None:
                    train_data = dataset[train_data_name]
                train_target_list = train_opt['target']
                _, train_x = train_data.get_spike_array()
                _, train_y = train_data.get_behavior_array(train_target_list)
                split_col, split_data = train_data.get_data_startswith(train_data.split_identifier)

                model_path = path.join(exp_path, 'model')
                os.makedirs(model_path)
                if len(split_col) == 0:
                    save_path = path.join(model_path, 'model')
                    model = train_pipeline(train_x, train_y, save_path, model_opt, logger)
                    model_list.append(model)
                else:
                    for idx, col in enumerate(split_col):
                        mask = split_data[:, idx]
                        save_path = path.join(model_path, f'model_{col}')
                        model = train_pipeline(train_x[mask == 1], train_y[mask == 1], save_path, model_opt, logger)
                        model_list.append(model)

        # test model
        logger.info('Test model.')
        test_data_name = test_opt.get('dataset')
        if test_data_name is not None:
            test_data = p_dataset.get(test_data_name)
            if test_data is None:
                test_data = dataset[test_data_name]
            test_target_list = test_opt['target']
            _, test_x = test_data.get_spike_array()
            test_y_col, test_y = test_data.get_behavior_array(test_target_list)
            split_col, split_data = test_data.get_data_startswith(test_data.split_identifier)
            label = {'col': test_y_col, 'name': split_col}
            if len(split_col) == 0:
                test_pipeline(test_x, test_y, model_list, metric_list)
            elif len(split_col) == len(model_list):
                test_pipeline(test_x, test_y, model_list, metric_list, label, split_data)

        logger.info(f'End experiment {exp_name}.')


def test_pipeline(test_x, test_y, model_list, metric_list, label=None, mask=None):
    metric_result = []
    for idx, model in enumerate(model_list):
        if mask is not None:
            result = model.validation(
                test_x[mask[:, idx] == 2], test_y[mask[:, idx] == 2], metric_list,
                label['col'], label['name'][idx]
            )
        else:
            result = model.validation(test_x, test_y, metric_list)
        metric_result.append(result)
    return metric_result


def train_pipeline(x, y, save_path, model_opt, logger):
    model = build_model(model_opt)
    logger.info('Start training.')
    start_time = time.time()
    model.fit(x, y)
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}.')
    logger.info('Save model.')
    model.save(save_path)
    return model


def processor_pipeline(dataset, processor_opt):
    p_dataset = {}
    for dataset_name, dataset_opt in processor_opt.items():
        is_copy = dataset_opt.get('is_copy', True)
        if is_copy:
            raise NotImplementedError
        else:
            nwb_dataset = dataset[dataset_name]
        for p_name, p_opt in dataset_opt.items():
            if isinstance(p_opt, dict):
                run_processor(nwb_dataset, p_opt)
        p_dataset[dataset_name] = nwb_dataset
    return p_dataset


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    run_pipeline(root)
