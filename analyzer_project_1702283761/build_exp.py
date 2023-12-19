import sys
import os
sys.path.append("../")
import streamlit as st
import yaml
from ndbox.utils.options import ordered_yaml
from ndbox.dataset import NWBDataset
from ndbox.utils import yaml_load
from ndbox.analyzer.sua import TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer
from ndbox.analyzer.sua import ISIAnalyzer, ISITimeHistAnalyzer, InstantFreqAnalyzer
from ndbox.analyzer.sua import PoincareMapAnalyzer, AutocorrelogramAnalyzer
from ndbox.analyzer.era import TuningAnalyzer

import pandas as pd
import numpy as np


def yaml_dump(opt: dict, fn: str):
    yaml_string = yaml.dump(opt, Dumper=ordered_yaml()[1])
    with open(fn, 'w') as f:
        f.write(yaml_string)
    return yaml_string


# 读取文件缓存: 数据文件的基本信息
cache = yaml_load('analyze_info/info.yml')
if cache is None:
    cache = {}
# 当前选取的数据集名称
dataset = None

# 已有的实验列表
experiments = os.listdir('experiments')
# 当前实验的配置信息
exp_opt = {
    'dataset': {},
    'experiment': {}
}


# 已有的功能列表
analyzer_list = {
    'Single Unit Analysis': [
        TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer, ISIAnalyzer,
        ISITimeHistAnalyzer, InstantFreqAnalyzer, PoincareMapAnalyzer, AutocorrelogramAnalyzer
    ],
    'Event related analysis': [
        TuningAnalyzer
    ],
    'Relative analysis': [

    ],
    'Multi unit analysis': [

    ]
}

def ce_select_dataset():
    # 1. some dataset to be chosen
    dataset_repo = {
        "sub-Indy": r"D:\datasets\SpikeData\nwb_sample\sub-Indy_desc-train_behavior+ecephys.nwb",
        "sub-Jenkins": r"D:\datasets\SpikeData\nwb_sample\sub-Jenkins_ses-medium_desc-train_behavior+ecephys.nwb",
        "sub-210862": r"D:\datasets\SpikeData\nwb_sample\sub-210862_ses-20130626_behavior+ecephys+ogen.nwb",
        "sub-219030": r"D:\datasets\SpikeData\nwb_sample\sub-219030_ses-20130901_behavior+ecephys+ogen.nwb",
        "sub-255201": r"D:\datasets\SpikeData\nwb_sample\sub-255201_ses-20141123_behavior+ecephys+ogen.nwb",
        "sub-257636": r"D:\datasets\SpikeData\nwb_sample\sub-257636_ses-20150331_behavior+ecephys+ogen.nwb"
    }
    dataset = st.selectbox('select a dataset', list(dataset_repo.keys()))
    exp_opt['dataset'] = {
        'name': dataset,
        'type': 'NWBDataset',
        'path': dataset_repo[dataset]
    }
    # 2. check if hit cache
    if dataset not in cache:
        current_dataset_info = {
            'name': dataset,
            'type': 'NWBDataset',
            'duration': None,
            'bin_size': None,
            'neuron_count': None,
            'min avg rates': None,
            'max avg rates': None,
        }
        nwb_data = NWBDataset(dataset_repo[dataset])
        behaviors = list(nwb_data.content_dict['behavior'].keys())
        behaviors = ['behavior/' + bh for bh in behaviors]
        current_dataset_info['bin_size'] = nwb_data.bin_size
        current_dataset_info['duration'] = nwb_data.data.shape[0] * nwb_data.bin_size
        current_dataset_info['neuron_count'] = nwb_data.spike_train.size
        firing_rates = [round(sp.size / current_dataset_info['duration'], 2) for sp in nwb_data.spike_train]
        current_dataset_info['min avg rates'] = min(firing_rates)
        current_dataset_info['max avg rates'] = max(firing_rates)
        cache[dataset] = {
            'info': current_dataset_info,
            'behaviors': behaviors,
            'firing_rates': firing_rates
        }
        yaml_dump(cache, 'analyze_info/info.yml')
    return dataset

def ce_input_exp_name():
    return st.text_input("name of the experiment")

def ce_select_analyzer_type():
    # 1. 加载已有的功能选项
    ana_type_list = list(analyzer_list.keys())
    ana_type = st.selectbox("Select one analyzer type", ana_type_list)
    st.text(ana_type)
    analyzers = analyzer_list[ana_type]
    ana_name_dict = {}
    ana_desc_dict = {}
    for a in analyzers:
        ana_name_dict[a.name] = a
        ana_desc_dict[a] = a.description
    ana_name = st.selectbox("Choose an analyzer", list(ana_name_dict.keys()))
    st.text(ana_desc_dict[ana_name_dict[ana_name]])
    exp_opt['experiment'] = {
        'type': ana_name_dict[ana_name].ana_type,
        'params_data': {},
        'params_plot': {}
    }

def neuron_selector():
    neuron_select = st.columns(2)
    firing_rates = cache[dataset]['firing_rates']
    dataset_info = cache[dataset]['info']
    mnr = min(firing_rates)
    mxr = max(firing_rates)
    with neuron_select[0]:
        sel = st.radio("Choose by firing rates or just the index",
                       ['firing rates', 'neuron index'])
    with neuron_select[1]:
        if sel == 'firing rates':
            st.text("Choose neurons have firing rates")
            ns_left = st.columns(2)
            with ns_left[0]:
                avg = (mnr + mxr) / 2
                min_fr = st.number_input("from(include)", min_value=mnr, step=0.5, max_value=mxr, value=avg)
            with ns_left[1]:
                max_fr = st.number_input("to(not include)", min_value=mnr, step=0.5, max_value=mxr, value=mxr)
        else:
            st.text("Choose neurons with index")
            ns_right = st.columns(2)
            with ns_right[0]:
                from_idx = st.number_input("from(include)", min_value=0, max_value=dataset_info['neuron_count'], step=1,
                                           value=int(dataset_info['neuron_count'] / 4))
            with ns_right[1]:
                to_idx = st.number_input("to(not include)", min_value=0, max_value=dataset_info['neuron_count'], step=1,
                                         value=int(dataset_info['neuron_count'] / 2))
    if sel == 'firing rates':
        checked_index = [i for i in range(len(firing_rates)) if min_fr < firing_rates[i] <= max_fr]
    else:
        checked_index = list(range(from_idx, to_idx))
    return checked_index

def ce_configure_params():
    if exp_opt['experiment']['type'] == TimeHistAnalyzer.ana_type:
        target = neuron_selector()
        exp_opt['experiment']['target'] = target
        st.text(f"You have chosen {len(target)} neurons.")
        ts = st.columns(3)
        duration = cache[dataset]['info']['duration']
        st.text('')
        st.text('Configure data process params:')
        with ts[0]:
            t_start = st.number_input("t_start", 0., duration, 0., 1.)
        with ts[1]:
            t_stop = st.number_input("t_stop", 0., duration, duration, 1.)
        with ts[2]:
            bin_size = st.number_input("bin_size", 0., 1., 0.005, 0.001, format='%.4f')
        ts = st.columns(2)
        with ts[0]:
            smooth = st.selectbox("smooth kernel", ['-', 'gas', 'avg'])
            if smooth == '-':
                smooth = None
                st.text("No smoothing method applied.")
            elif smooth == 'gas':
                st.text("Gaussian kernel applied.")
            elif smooth == 'avg':
                st.text("Average kernel applied.")
        with ts[1]:
            kernel_size = st.number_input("kernel window size", 0, 50, 3, 1)
            st.text("The bigger the kernel size, the smoother the histogram.")
        exp_opt['experiment']['params_data']['t_start'] = t_start
        exp_opt['experiment']['params_data']['t_stop'] = t_stop
        exp_opt['experiment']['params_data']['bin_size'] = bin_size
        exp_opt['experiment']['params_data']['smooth'] = smooth
        exp_opt['experiment']['params_data']['kernel_size'] = kernel_size
        st.text('')
        st.text('Configure plot params')
        form = st.selectbox('form', ['bar', 'step'])
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['form'] = form
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == CumActivityAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == RasterAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == ISIAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == ISITimeHistAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == InstantFreqAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == PoincareMapAnalyzer.ana_type:
        pass
    elif exp_opt['experiment']['type'] == AutocorrelogramAnalyzer.ana_type:
        pass


def config_exp():
    st.subheader("1. Select a dataset.")
    global dataset
    dataset = ce_select_dataset()
    st.subheader("2. Input a experiment name")
    exp_name = ce_input_exp_name()
    st.subheader("3. Select an analyzer type")
    ce_select_analyzer_type()
    st.subheader("4. Configure the experiment params")
    ce_configure_params()
    clk = st.button("Build Experiment")
    if clk:
        exp_path = os.path.join('experiments', exp_name)
        if exp_name == '' or exp_name is None:
            st.error("Please input a valid experiment name.")
        elif not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            st.error(f"{exp_name} already exist, if continue, original data will be covered!")
        yaml_dump(exp_opt, os.path.join(exp_path, 'config.yml'))
    return exp_name

def execute_exp(exp_name):
    os.system('python run_exp.py -e '+exp_name)


def nav_dataset_info():
    if cache is not None and dataset in cache:
        current_dataset_info = cache[dataset]['info']
        bi_df = pd.DataFrame(index=list(current_dataset_info.keys()), columns=['values'],
                             data=list(current_dataset_info.values()))
        st.dataframe(bi_df)

def nav_experiment_config():
    st.json(exp_opt)

def navigate():
    st.header("Preview your analyze information here")
    st.subheader("Dataset information")
    nav_dataset_info()
    st.subheader("Experiment configuration")
    nav_experiment_config()

def main_ui():
    st.title("A simple spike data processing helper -- NDToolbox")
    st.header('Build your experiment')
    exp_name = config_exp()
    with st.sidebar:
        navigate()
    st.header('Execute your experiment')
    clk = st.button("Execute")
    if clk:
        execute_exp(exp_name)
        # show result
        image_dir = os.path.join('experiments', exp_name, 'result')
        image_names = os.listdir(image_dir)
        dir_list = [os.path.join(image_dir, im) for im in image_names]

        im_col2 = st.columns(2)
        left_half = int(len(dir_list) / 2)
        with im_col2[0]:
            for ii in range(left_half):
                st.image(dir_list[ii], caption=image_names[ii])
        with im_col2[1]:
            for ii in range(left_half, len(dir_list)):
                st.image(dir_list[ii], caption=image_names[ii])


if __name__ == '__main__':
    main_ui()

