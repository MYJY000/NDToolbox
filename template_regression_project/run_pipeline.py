import os
import random
import argparse
import time
import datetime

import numpy as np
import pandas as pd
from os import path
from collections import OrderedDict

from ndbox.model import build_model
from ndbox.processor import TRAIN_MASK, TEST_MASK
from ndbox.utils import (yaml_load, opt2str, set_random_seed, get_root_logger,
                         load_image, restore_image)

try:
    import user_define_modules
except ImportError:
    pass


def parse_options():
    parser = argparse.ArgumentParser()
    default_config = path.join(root, 'config.yml')
    parser.add_argument('-opt', type=str, default=default_config, help='Config path.')
    args = parser.parse_args()
    opt = yaml_load(args.opt)

    opt['path'] = {}
    name = opt.get('name', 'result')
    temp_path = path.join(root, 'temp_files')
    result_path = path.join(root, 'results', f'{name}_{int(time.time())}')
    log_path = path.join(result_path, 'log.txt')

    opt['path']['root'] = root
    opt['path']['temp'] = temp_path
    opt['path']['result'] = result_path
    opt['path']['log'] = log_path
    os.makedirs(result_path)

    seed = opt.get('random_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['random_seed'] = seed
    set_random_seed(seed)

    return opt, args


def run_pipeline():
    opt, args = parse_options()
    logger = get_root_logger(log_file=opt['path']['log'])
    logger.info(opt2str(opt))
    result_path = str(opt['path']['result'])
    temp_path = opt['path']['temp']

    datasets_opt = opt.get('dataset', OrderedDict())
    experiments_opt = opt.get('experiment', OrderedDict())

    # load dataset
    datasets = OrderedDict()
    for dataset_name, dataset_opt in datasets_opt.items():
        datasets[dataset_name] = load_image(temp_path, dataset_opt)

    for index, (exp_name, exp_opt) in enumerate(experiments_opt.items()):
        exp_path = path.join(result_path, exp_name)
        os.makedirs(exp_path)
        logger.info(f'Start experiment {exp_name}.')

        processor_opt = exp_opt.get('processor', OrderedDict())
        model_opt = exp_opt.get('model', OrderedDict())
        train_opt = exp_opt.get('train', OrderedDict())
        val_opt = exp_opt.get('val', OrderedDict())
        test_list = [item for item in exp_opt.keys() if str(item).startswith('test')]
        metrics_opt = exp_opt.get('metrics', OrderedDict())

        # processor dataset
        for dataset_name in processor_opt.keys():
            restore_image(
                temp_path=temp_path,
                nwb_data=datasets[dataset_name],
                dataset_opt=datasets_opt[dataset_name],
                processor_opt=processor_opt[dataset_name]
            )

        # fit model
        model_load_path = model_opt.get('path')
        model_list = []
        if model_load_path is not None:
            model = build_model(model_opt)
            if model.identifier == 'ML':
                model.load(model_load_path)
            elif model.identifier == 'DL':
                '''
                1. pretrain_net, load_network
                2. state_path, load_state
                '''
                pass
            model_list.append(model)
        else:
            train_dataset_name = train_opt.get('dataset')
            val_dataset_name = val_opt.get('dataset')
            if train_dataset_name is not None:
                train_dataset = datasets[train_dataset_name]
                train_target = train_opt['target']
                # prepare data
                _, train_x = train_dataset.get_spike_array()
                train_y_col, train_y = train_dataset.get_behavior_array(train_target)
                split_col, split_data = train_dataset.get_data_startswith(train_dataset.split_identifier)
                val_x = None
                val_y = None
                if val_dataset_name is not None:
                    val_dataset = datasets[val_dataset_name]
                    val_target = val_opt['target']
                    val_x = val_dataset.get_spike_array()
                    val_y = val_dataset.get_behavior_array(val_target)

                # fit & save
                model_path = path.join(exp_path, 'model')
                os.makedirs(model_path)
                if len(split_col) == 0:  # all data
                    model = train_pipeline(
                        train_x=train_x,
                        train_y=train_y,
                        train_opt=train_opt,
                        model_path=model_path,
                        model_name_suffix='',
                        model_opt=model_opt,
                        val_x=val_x,
                        val_y=val_y,
                        val_opt=val_opt,
                        logger=logger
                    )
                    model_list.append(model)
                else:
                    for idx, col in enumerate(split_col):
                        mask = split_data[:, idx]
                        model = train_pipeline(
                            train_x=train_x[mask == TRAIN_MASK],
                            train_y=train_y[mask == TRAIN_MASK],
                            train_opt=train_opt,
                            model_path=model_path,
                            model_name_suffix='_' + str(col),
                            model_opt=model_opt,
                            val_x=val_x,
                            val_y=val_y,
                            val_opt=val_opt,
                            logger=logger,
                        )
                        model_list.append(model)
                # calculate and save train metrics
                metrics_result = None
                if len(split_col) == 0:
                    metrics_result = test_pipeline(
                        x=train_x,
                        y_true=train_y,
                        model_list=model_list,
                        metrics_opt=metrics_opt,
                    )
                elif len(split_col) == len(model_list):
                    metrics_result = test_pipeline(
                        x=train_x,
                        y_true=train_y,
                        model_list=model_list,
                        metrics_opt=metrics_opt,
                        mask=split_data,
                        select=[TRAIN_MASK]
                    )
                if metrics_result is not None:
                    save_metrics(
                        save_path=path.join(exp_path, 'train_metrics.csv'),
                        metrics_result=metrics_result,
                        columns_name=train_y_col,
                        metrics_opt=metrics_opt
                    )

        # test model
        logger.info('Test model.')
        for item in test_list:
            test_opt = exp_opt[item]
            test_dataset_name = test_opt.get('dataset')
            if test_dataset_name is not None:
                test_dataset = datasets[test_dataset_name]
                test_target = test_opt['target']
                _, test_x = test_dataset.get_spike_array()
                test_y_col, test_y = test_dataset.get_behavior_array(test_target)
                split_col, split_data = test_dataset.get_data_startswith(test_dataset.split_identifier)
                metrics_result = None
                if len(split_col) == 0:
                    metrics_result = test_pipeline(
                        x=test_x,
                        y_true=test_y,
                        model_list=model_list,
                        metrics_opt=metrics_opt,
                    )
                elif len(split_col) == len(model_list):
                    metrics_result = test_pipeline(
                        x=test_x,
                        y_true=test_y,
                        model_list=model_list,
                        metrics_opt=metrics_opt,
                        mask=split_data,
                        select=[TEST_MASK]
                    )
                else:
                    logger.error(f'Test model failed. Split columns [{len(split_col)}] not equal'
                                 f' to number of models [{len(model_list)}].')
                # save test metrics
                if metrics_result is not None:
                    save_metrics(
                        save_path=path.join(exp_path, f'{item}_{test_dataset_name}_metrics.csv'),
                        metrics_result=metrics_result,
                        columns_name=test_y_col,
                        metrics_opt=metrics_opt
                    )

        # recover dataset
        if index < len(experiments_opt) - 1:
            for dataset_name in processor_opt.keys():
                restore_image(
                    temp_path=temp_path,
                    nwb_data=datasets[dataset_name],
                    dataset_opt=datasets_opt[dataset_name]
                )

        logger.info(f'End experiment {exp_name}.')


def train_pipeline(train_x, train_y, train_opt, model_path, model_name_suffix, model_opt,
                   val_x, val_y, val_opt, logger):
    model = build_model(model_opt)
    model.name = model.name + model_name_suffix
    logger.info('Start training.')
    start_time = time.time()
    if model.identifier == 'ML':
        model.fit(train_x, train_y)
    elif model.identifier == 'DL':
        '''
        1. model.feed_data(data)
            data = {
                'train_x': train_x,
                'train_y': train_y,
                'train_opt': train_opt,
                'val_x': val_x,
                'val_y': val_y,
                'val_opt': val_opt,
            }
        2. init train setting
        3. loop train epoch
        '''
        pass
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}.')
    logger.info('Save model.')
    save_path = path.join(model_path, model.name)
    model.save(save_path)
    return model


def test_pipeline(x, y_true, model_list, metrics_opt, mask=None, select=None):
    metrics_result = {}
    for idx, model in enumerate(model_list):
        if mask is not None:
            _mask = np.isin(mask[:, idx], select)
            metric_result = model.validation(x[_mask], y_true[_mask], metrics_opt)
        else:
            metric_result = model.validation(x, y_true, metrics_opt)
        metrics_result[model.name] = metric_result
    return metrics_result


def save_metrics(save_path, metrics_result, columns_name, metrics_opt):
    export_metric = []
    df_index = []
    metrics_name = str(list(metrics_opt.keys()))
    df_col = [str(col) + metrics_name for col in columns_name]
    for model_name, metric_result in metrics_result.items():
        df_index.append(model_name)
        zip_data = zip(*metric_result.values())
        row_data = [list(col) for col in zip_data]
        export_metric.append(row_data)
    df = pd.DataFrame(export_metric, columns=df_col, index=df_index)
    df.to_csv(save_path)


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    run_pipeline()
