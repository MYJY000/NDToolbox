import os
import argparse
import time
from os import path

from ndbox.utils import create_directory_and_files


def parse_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='-', help='Project name.')
    parser.add_argument('-mode', type=str, default='regression',
                        help="Project mode. Default 'regression'. "
                             "'r' is regression, "
                             "'c' is classification, "
                             "'a' is analyzer.")
    parser.add_argument('-path', type=str, default=root_path, help='Project path.')
    args = parser.parse_args()

    trans = {'r': 'regression', 'c': 'classification', 'a': 'analyzer'}
    args.mode = args.mode if trans.get(args.mode) is None else trans.get(args.mode)

    if args.name == '-':
        args.name = args.mode + '_project_' + str(int(time.time()))

    if args.mode == 'regression':
        create_regression_project(args)
    elif args.mode == 'classification':
        create_classification_project(args)
    elif args.mode == 'analyzer':
        create_analyzer_project(args)
    else:
        raise NotImplementedError(f'Mode {args.mode} not support!')


def create_regression_project(args):
    project_path, user_define_path = create_project_structure(path.join(args.path, args.name))
    regression_pipeline_init(project_path)
    regression_web_ui_init(project_path)


def create_classification_project(args):
    pass


def create_analyzer_project(args):
    # 1. create analyzer project structure
    project_path = path.join(args.path, args.name)
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'experiments': {},      # each experiment will be carried here
        'analyze_info': {
            'info.yml': None    # cache the basic info about the dataset
        },
        'main_ui.py': None,
        'main.py': None
    }
    create_directory_and_files(project_path, dir_structure)

    from ndbox import analyzer
    ana_path = os.path.abspath(analyzer.__file__)
    ana_path_list = ana_path.split(os.sep)

    # 2. initialize the offline pipeline
    pipeline_path = path.join(project_path, 'run_exp.py')
    ana_path_list[-1] = analyzer.pipeline_path
    analyzer_init(ana_path_list, pipeline_path)
    # 3. initialize the ui pipeline
    pipeline_ui_path = path.join(project_path, 'build_exp.py')
    ana_path_list[-1] = analyzer.pipeline_ui_path
    analyzer_init(ana_path_list, pipeline_ui_path)


def analyzer_init(src, desc):
    src_path = os.sep.join(src)
    with open(src_path, 'r') as sf:
        src_content = sf.read()
    with open(desc, 'w') as df:
        df.write(src_content)


def create_project_structure(project_path):
    if path.exists(project_path):
        raise FileExistsError(f"Project {project_path} already exists!")
    os.makedirs(project_path)
    dir_structure = {
        'user_define_modules': {
            '__init__.py': None
        },
        'temp_files': {},
        'config.yml': None,
        'web_ui.py': None,
        'result_report.py': None,
        'run_pipeline.py': None
    }
    create_directory_and_files(project_path, dir_structure)
    user_define_path = path.join(project_path, 'user_define_modules')
    user_define_init(path.join(user_define_path, '__init__.py'))
    return project_path, user_define_path


def user_define_init(filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            """import glob
import importlib
from os import path


py_files = glob.glob(path.join(path.dirname(__file__), '*.py'))
module_names = [path.splitext(path.basename(py_file))[0] for py_file in py_files]
modules = [importlib.import_module(f'user_define_modules.{module_name}') for module_name in module_names]
"""
        )
        f.close()


def regression_pipeline_init(project_path):
    filepath = path.join(project_path, 'run_pipeline.py')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            """import datetime
import os
import random
import argparse
import time
from os import path

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
        is_copy = dataset_opt.get('is_copy', False)
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
"""
        )
        f.close()


def regression_web_ui_init(project_path):
    filepath = path.join(project_path, 'web_ui.py')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            """import inspect
import os
import yaml
import streamlit as st
from copy import deepcopy
from collections import OrderedDict

from ndbox.dataset import NWBDataset
from ndbox.utils import PROCESSOR_REGISTRY, MODEL_REGISTRY, METRIC_REGISTRY
from ndbox.utils.options import ordered_yaml


def dict2yaml(ordered_dict):
    yaml_string = yaml.dump(ordered_dict, Dumper=ordered_yaml()[1])
    return yaml_string


def yaml2dict(yaml_string):
    ordered_dict = yaml.load(yaml_string, Loader=ordered_yaml()[0])
    return ordered_dict


def general_setting():
    opt = st.session_state.config_dict
    st.markdown('# General Setting')
    name = st.text_input('name', value=opt.get('name'))
    random_seed = st.number_input('random_seed', min_value=0, step=1, value=opt.get('random_seed'))
    if st.button('Apply') and name:
        opt['name'] = name
        opt['random_seed'] = random_seed
        st.session_state.config_dict = opt
    st.sidebar.json(st.session_state.config_dict)


def dataset_setting():
    opt = st.session_state.config_dict
    st.markdown('# Dataset Setting')
    new_dataset_name = st.text_input('New Dataset Name')
    if st.button('Add Dataset') and new_dataset_name:
        opt['dataset'][new_dataset_name] = {
            'name': new_dataset_name
        }
        st.session_state.config_dict = opt
        st.session_state.dataset_list[new_dataset_name] = add_dataset
    select_dataset = st.selectbox('Select Dataset',
                                  list(st.session_state.dataset_list.keys()) + ['-'])
    if select_dataset in st.session_state.dataset_list:
        st.session_state.dataset_list[select_dataset](select_dataset)
    st.sidebar.json(st.session_state.config_dict)


def add_dataset(dataset_name):
    opt = st.session_state.config_dict['dataset'][dataset_name]
    if dataset_name not in st.session_state.tmp_data:
        st.session_state.tmp_data[dataset_name] = {}
    tmp = st.session_state.tmp_data[dataset_name]
    file_type = st.radio('type', ['NWBDataset'])
    file_path = st.text_input('path', value=opt.get('path'))
    if file_path and file_type:
        try:
            nwb_data = tmp.get(file_path)
            if nwb_data is None:
                nwb_data = NWBDataset(file_path)
                st.session_state.tmp_data[dataset_name][file_path] = nwb_data
            preview_dict = deepcopy(nwb_data.content_dict)
            preview_dict = nwb_preview_tool(nwb_data.data_dict, preview_dict)
            st.json(preview_dict)
            opt['type'] = file_type
            opt['path'] = file_path
            behavior_select = st.multiselect(
                'Behavior Field Select',
                list(nwb_data.data_dict.keys()),
                default=opt.get('behavior')
            )
            opt['behavior'] = behavior_select
            st.session_state.config_dict['dataset'][dataset_name] = opt
        except Exception as e:
            st.error('File not support!\\n' + str(e))


def nwb_preview_tool(data_dict, preview_dict, prefix=''):
    if not isinstance(preview_dict, dict):
        return preview_dict
    tmp_dict = preview_dict
    for key, value in preview_dict.items():
        name = prefix + key
        if not isinstance(value, dict) or len(value) == 0:
            msg = data_dict.get(name)
            if msg is not None:
                tmp_dict[key] = str(msg)
            else:
                tmp_dict[key] = 'No description.'
        else:
            tmp_dict[key] = nwb_preview_tool(data_dict, value, name + '/')
    return tmp_dict


def experiment_setting():
    opt = st.session_state.config_dict
    st.markdown('# Experiment Setting')
    new_exp_name = st.text_input('New Experiment Name')
    if st.button('Add Experiment') and new_exp_name:
        opt['experiment'][new_exp_name] = {
            'name': new_exp_name
        }
        st.session_state.config_dict = opt
        st.session_state.experiment_list[new_exp_name] = add_experiment
    select_experiment = st.selectbox('Select Experiment',
                                     list(st.session_state.experiment_list.keys()) + ['-'])
    if select_experiment in st.session_state.experiment_list:
        st.session_state.experiment_list[select_experiment](select_experiment)
    st.sidebar.json(st.session_state.config_dict)


def add_experiment(experiment_name):
    opt = st.session_state.config_dict['experiment'][experiment_name]
    tab_processor, tab_model, tab_train_test, tab_metrics = st.tabs(
        ['processor', 'model', 'train_test', 'metrics']
    )
    opt1 = opt.get('processor', {})
    opt2 = opt.get('model', {})
    opt3 = opt.get('train', {})
    opt4 = opt.get('test', {})
    opt5 = opt.get('metrics', {})

    with tab_processor:
        select_dataset = st.selectbox('Select Dataset',
                                      list(st.session_state.dataset_list.keys()) + ['-'])
        if select_dataset in st.session_state.dataset_list:
            if select_dataset not in opt1:
                opt1[select_dataset] = {}
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                r_list = ['resample', '-']
                r_index = r_list.index(opt1[select_dataset].get('1_resample', {}).get('type', '-'))
                r_type = st.selectbox('resample', r_list, index=r_index)
                if r_type and r_type != '-':
                    if '1_resample' not in opt1[select_dataset]:
                        opt1[select_dataset]['1_resample'] = {}
                    opt1[select_dataset]['1_resample']['type'] = r_type
                    param_dict = get_param_dict(PROCESSOR_REGISTRY, r_type, ignore=['nwb_data'])
                    for key, value in param_dict.items():
                        default_value = opt1[select_dataset]['1_resample'].get(key, value)
                        v = st.text_input(label=key, value=default_value)
                        opt1[select_dataset]['1_resample'][key] = v
                else:
                    if '1_resample' in opt1[select_dataset]:
                        opt1[select_dataset].pop('1_resample')
            with col2:
                sm_list = ['gaussian_smooth', '-']
                sm_index = sm_list.index(opt1[select_dataset].get('2_smooth', {}).get('type', '-'))
                sm_type = st.selectbox('smooth', sm_list, index=sm_index)
                if sm_type and sm_type != '-':
                    if '2_smooth' not in opt1[select_dataset]:
                        opt1[select_dataset]['2_smooth'] = {}
                    opt1[select_dataset]['2_smooth']['type'] = sm_type
                    param_dict = get_param_dict(PROCESSOR_REGISTRY, sm_type, ignore=['nwb_data'])
                    for key, value in param_dict.items():
                        default_value = opt1[select_dataset]['2_smooth'].get(key, value)
                        v = st.text_input(label=key, value=default_value)
                        opt1[select_dataset]['2_smooth'][key] = v
                else:
                    if '2_smooth' in opt1[select_dataset]:
                        opt1[select_dataset].pop('2_smooth')
            with col3:
                l_list = ['lag_offset', '-']
                l_index = l_list.index(opt1[select_dataset].get('3_lag', {}).get('type', '-'))
                l_type = st.selectbox('lag', l_list, index=l_index)
                if l_type and l_type != '-':
                    if '3_lag' not in opt1[select_dataset]:
                        opt1[select_dataset]['3_lag'] = {}
                    opt1[select_dataset]['3_lag']['type'] = l_type
                    param_dict = get_param_dict(PROCESSOR_REGISTRY, l_type, ignore=['nwb_data'])
                    for key, value in param_dict.items():
                        default_value = opt1[select_dataset]['3_lag'].get(key, value)
                        v = st.text_input(label=key, value=default_value)
                        opt1[select_dataset]['3_lag'][key] = v
                else:
                    if '3_lag' in opt1[select_dataset]:
                        opt1[select_dataset].pop('3_lag')
            with col4:
                sp_list = ['train_test_bins_split', 'KFord_split', '-']
                sp_index = sp_list.index(opt1[select_dataset].get('4_split', {}).get('type', '-'))
                sp_type = st.selectbox('split', sp_list, index=sp_index)
                if sp_type and sp_type != '-':
                    if '4_split' not in opt1[select_dataset]:
                        opt1[select_dataset]['4_split'] = {}
                    opt1[select_dataset]['4_split']['type'] = sp_type
                    param_dict = get_param_dict(PROCESSOR_REGISTRY, sp_type, ignore=['nwb_data'])
                    for key, value in param_dict.items():
                        default_value = opt1[select_dataset]['4_split'].get(key, value)
                        v = st.text_input(label=key, value=default_value)
                        opt1[select_dataset]['4_split'][key] = v
                else:
                    if '4_split' in opt1[select_dataset]:
                        opt1[select_dataset].pop('4_split')

    with tab_model:
        model_list = list(MODEL_REGISTRY.keys()) + ['-']
        model_index = model_list.index(opt2.get('type', model_list[0]))
        select_model = st.selectbox('Select Model', model_list, index=model_index)
        if select_model != '-':
            opt2['type'] = select_model
            param_dict = get_param_dict(MODEL_REGISTRY, select_model)
            for key, value in param_dict.items():
                default_value = opt2.get(key, value)
                v = st.text_input(label=key, value=default_value)
                opt2[key] = v
        else:
            opt2 = {}

    with tab_train_test:
        train_col, test_col = st.columns([1, 1])
        with train_col:
            train_list = list(st.session_state.dataset_list.keys()) + ['-']
            train_index = train_list.index(opt3.get('dataset', train_list[0]))
            train_dataset = st.selectbox('Train Dataset', train_list, index=train_index)
            if train_dataset in st.session_state.dataset_list:
                opt3['dataset'] = train_dataset
                train_target = []
                behavior_list = st.session_state.config_dict['dataset'].get(train_dataset, {}).get('behavior', [])
                for item in opt3.get('target', []):
                    if item in behavior_list:
                        train_target.append(item)
                train_target_select = st.multiselect(
                    'Train Target Select',
                    behavior_list,
                    default=train_target
                )
                opt3['target'] = train_target_select
            else:
                opt3 = {}
        with test_col:
            test_list = list(st.session_state.dataset_list.keys())
            test_index = test_list.index(opt4.get('dataset', test_list[0]))
            test_dataset = st.selectbox('Test Dataset', test_list, index=test_index)
            if test_dataset in st.session_state.dataset_list:
                opt4['dataset'] = test_dataset
                test_target = []
                behavior_list = st.session_state.config_dict['dataset'].get(test_dataset, {}).get('behavior', [])
                for item in opt4.get('target', []):
                    if item in behavior_list:
                        test_target.append(item)
                test_target_select = st.multiselect(
                    'Test Target Select',
                    behavior_list,
                    default=test_target
                )
                opt4['target'] = test_target_select
            else:
                opt4 = {}

    with tab_metrics:
        metrics_list = list(METRIC_REGISTRY.keys())
        selected_list = []
        for item in opt5.values():
            selected_list.append(item['type'])
        metrics_select = st.multiselect(
            'Metrics Select',
            metrics_list,
            default=selected_list
        )
        for item in metrics_select:
            opt5[item] = {'type': item}

    if st.button('Apply'):
        opt['processor'] = opt1
        opt['model'] = opt2
        opt['train'] = opt3
        opt['test'] = opt4
        opt['metrics'] = opt5
        st.session_state.config_dict['experiment'][experiment_name] = opt


def get_param_dict(registry, fun_type, ignore=None):
    fun = registry.get(fun_type)
    param_list = inspect.signature(fun).parameters
    ret_dict = {}
    for name, parameter in param_list.items():
        if ignore is not None and str(name) in ignore:
            continue
        if name == 'args' or name == 'kwargs':
            continue
        if parameter.default is inspect.Parameter.empty:
            ret_dict[name] = None
        else:
            ret_dict[name] = str(parameter.default)
    return ret_dict


def web_ui():
    if 'config_dict' not in st.session_state:
        st.session_state.config_dict = OrderedDict({
            'name': 'nd',
            'random_seed': 0,
            'dataset': {},
            'experiment': {}
        })
    if 'dataset_list' not in st.session_state:
        st.session_state.dataset_list = {}
    if 'experiment_list' not in st.session_state:
        st.session_state.experiment_list = {}
    if 'tmp_data' not in st.session_state:
        st.session_state.tmp_data = {}

    page_names_to_funcs = {
        'General': general_setting,
        'Dataset': dataset_setting,
        'Experiment': experiment_setting
    }

    page_name = st.sidebar.selectbox('Select Box', page_names_to_funcs.keys())
    st.sidebar.text('Config')
    page_names_to_funcs[page_name]()

    export_file = st.sidebar.text_input('Export Path', value='config.yml')
    if st.sidebar.button('Export Config') and export_file:
        config_str = dict2yaml(st.session_state.config_dict)
        config_str = config_str.replace('\\'', '')
        root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
        config_path = os.path.join(root_path, export_file)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_str)
            f.close()
            st.sidebar.success('Export Success')


if __name__ == '__main__':
    web_ui()
"""
        )
        f.close()


if __name__ == '__main__':
    root = path.abspath(path.join(__file__, path.pardir))
    parse_options(root)
