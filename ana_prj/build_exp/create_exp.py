import streamlit as st
from st_pages import add_page_title
import time
from app import cache_load, dump_yml, cache_dump
from dataset_loader import dataset_repo
import os
import sys

sys.path.append("../../")
from ndbox.analyzer.sua import TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer
from ndbox.analyzer.sua import ISIAnalyzer, ISITimeHistAnalyzer, InstantFreqAnalyzer
from ndbox.analyzer.sua import PoincareMapAnalyzer, AutocorrelogramAnalyzer
from ndbox.analyzer.era import TuningAnalyzer, PeriStimulusAnalyzer
from ndbox.analyzer.rta import CrossCorrelationAnalyzer, JointPSTHAnalyzer

# 全局的实验配置
cache = cache_load()
dataset = cache['last_dataset']
exp_opt = {
    'dataset': {
        'name': dataset,
        'type': 'NWBDataset',
        'path': dataset_repo[dataset]
    },
    'experiment': {}
}
behaviors = cache[dataset]['behaviors']
events = cache[dataset]['events']
# 实验名称
add_page_title(layout="wide")
st.subheader("Input experiment's name(or generate one)")
exp_name = st.text_input("experiment name")

# 分析类型
st.subheader("Choose one analysis")
# 已有的功能列表
analyzer_list = {
    'Single Unit Analysis': [
        TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer, ISIAnalyzer,
        ISITimeHistAnalyzer, InstantFreqAnalyzer, PoincareMapAnalyzer, AutocorrelogramAnalyzer
    ],
    'Event related analysis': [
        TuningAnalyzer, PeriStimulusAnalyzer
    ],
    'Relative analysis': [
        CrossCorrelationAnalyzer, JointPSTHAnalyzer
    ],
    'Multi unit analysis': [

    ]
}
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
duration = cache[dataset]['info']['duration']

# 参数设置
st.subheader("Analysis params configure")

def shift_p():
    shp = st.selectbox("Shift predictor", ['-', 'random', 'average'])
    if shp == '-':
        shp = None
    exp_opt['experiment']['params_data']['shift_predictor'] = shp

def neuron_selector(key=False):
    if not key:
        keys = [1, 2, 3, 4, 5]
    else:
        keys = [6, 7, 8, 9, 10]
    neuron_select = st.columns(2)
    firing_rates = cache[dataset]['firing_rates']
    dataset_info = cache[dataset]['info']
    mnr = min(firing_rates)
    mxr = max(firing_rates)
    with neuron_select[0]:
        sel = st.radio("Choose by firing rates or just the index",
                       ['firing rates', 'neuron index'], key=keys[0])
    with neuron_select[1]:
        if sel == 'firing rates':
            st.text("Choose neurons have firing rates")
            ns_left = st.columns(2)
            with ns_left[0]:
                avg = (mnr + mxr) / 2
                min_fr = st.number_input("from(include)", min_value=mnr, step=0.5,
                                         max_value=mxr, value=avg, key=keys[1])
            with ns_left[1]:
                max_fr = st.number_input("to(not include)", min_value=mnr, step=0.5,
                                         max_value=mxr, value=mxr, key=keys[2])
        else:
            st.text("Choose neurons with index")
            ns_right = st.columns(2)
            with ns_right[0]:
                from_idx = st.number_input("from(include)", min_value=0, max_value=dataset_info['neuron_count'], step=1,
                                           value=int(dataset_info['neuron_count'] / 4), key=keys[3])
            with ns_right[1]:
                to_idx = st.number_input("to(not include)", min_value=0, max_value=dataset_info['neuron_count'], step=1,
                                         value=int(dataset_info['neuron_count'] / 2), key=keys[4])
    if sel == 'firing rates':
        checked_index = [i for i in range(len(firing_rates)) if min_fr < firing_rates[i] <= max_fr]
    else:
        checked_index = list(range(from_idx, to_idx))
    return checked_index


def t_start_stop():
    ts = st.columns(2)
    with ts[0]:
        t_start = st.number_input("t_start", 0., duration, 0., 1.)
    with ts[1]:
        t_stop = st.number_input("t_stop", 0., duration, duration, 1.)
    exp_opt['experiment']['params_data']['t_start'] = t_start
    exp_opt['experiment']['params_data']['t_stop'] = t_stop


def bias_start_stop():
    ts = st.columns(2)
    with ts[0]:
        bias_start = st.number_input("bias_start", 0.001, 2.5, 0.2, 0.05, format='%.4f')
    with ts[1]:
        bias_stop = st.number_input("bias_stop", 0.001, 2.5, 0.2, 0.05, format='%.4f')
    exp_opt['experiment']['params_data']['bias_start'] = bias_start
    exp_opt['experiment']['params_data']['bias_stop'] = bias_stop


def t_start_stop_bs():
    ts = st.columns(3)
    with ts[0]:
        t_start = st.number_input("t_start", 0., duration, 0., 1.)
    with ts[1]:
        t_stop = st.number_input("t_stop", 0., duration, duration, 1.)
    with ts[2]:
        bin_size = st.number_input("bin_size", 0., 5., 0.005, 0.001, format='%.4f')
    exp_opt['experiment']['params_data']['t_start'] = t_start
    exp_opt['experiment']['params_data']['t_stop'] = t_stop
    exp_opt['experiment']['params_data']['bin_size'] = bin_size


def min_max_width():
    ts = st.columns(2)
    with ts[0]:
        min_width = st.number_input("min_width", 0., 0.1, 0., 0.001, format='%.4f')
    with ts[1]:
        max_width = st.number_input("max_width", 0., 1., 0.1, 0.001, format='%.4f')
    exp_opt['experiment']['params_data']['min_width'] = min_width
    exp_opt['experiment']['params_data']['max_width'] = max_width


def smooth_kernel():
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
    exp_opt['experiment']['params_data']['smooth'] = smooth
    exp_opt['experiment']['params_data']['kernel_size'] = kernel_size


def neuron_targets(refer=False):
    target = neuron_selector(refer)
    if not refer:
        exp_opt['experiment']['target'] = target
    else:
        exp_opt['experiment']['refer'] = target
    st.text(f"You have chosen {len(target)} neurons.")


def event_picker():
    # 选择 event 所在的列名或者人工指定一系列的时间戳
    e_name_list = list(events.keys())
    how_ev = st.radio("Choose an event type or input event timestamp manually.",
                      ["Load event type", "Input event timestamp manually"])
    if how_ev == "Load event type":
        event_type = st.selectbox("Choose an event", e_name_list)
        event_stamp = events[event_type]
        n = 16
        st.text(f"Event {event_type} has {len(event_stamp)} time picks.")
        st.text(f"You may just choose a few of them (No more than {n} is a good choice).")
        ts = st.columns(2)
        with ts[0]:
            from_e_id = st.number_input("from: ", 0, len(event_stamp), 0, 1)
        with ts[1]:
            val = n if len(event_stamp) > n else len(event_stamp)
            to_e_id = st.number_input("to: ", 0, len(event_stamp), val, 2)
        if from_e_id != 0 or to_e_id != len(event_stamp):
            st.text(f"You choose {from_e_id}-th to {to_e_id}-th time picks from "
                    f"event {event_type}\nFor convenient, you can give it a new name")
            event_type = st.text_input("New name", "event_sampled_default")
            events[event_type] = [event_stamp[e] for e in range(from_e_id, to_e_id)]
            cache[dataset]['events'] = events
    else:
        st.text("You can specify an event type manually")
        duration = cache[dataset]['info']['duration']
        bias = duration / 10
        st.text("Existing event list:")
        st.text(e_name_list)
        st.text("")
        event_type = st.text_input("New event name", "event_01")
        ev_str = st.text_input("Input a series of timestamps, use `,` to separate",
                               f"{bias * 0.2}, {bias * 0.7}, {bias * 1.2}, {bias * 2}, {bias * 2.6}, "
                               f"{bias * 5.4}, {bias * 6.3}, {bias * 7}, {bias * 8}")
        if st.button("Add"):
            events[event_type] = [float(word.strip()) for word in ev_str.split(',')]
            cache[dataset]['events'] = events
    if event_type in events:
        st.text("Your choice:")
        st.text(f"Event: {event_type}")
        st.text(f"Event train: {events[event_type]}")
        exp_opt['experiment']['events'] = events[event_type]

# 选择
def ce_configure_params():
    if exp_opt['experiment']['type'] == TimeHistAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop_bs()
        smooth_kernel()
        st.text('')
        st.text('Configure plot params')
        form = st.selectbox('form', ['bar', 'step'])
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['form'] = form
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == CumActivityAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop_bs()
        st.text('')
        st.text('Configure plot params')
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == RasterAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop()
        st.text('')
        st.text('Configure plot params')
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == ISIAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop_bs()
        min_max_width()
        st.text('')
        st.text('Configure plot params')
        form = st.selectbox('form', ['bar', 'step'])
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['form'] = form
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == ISITimeHistAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop()
        min_max_width()
        st.text('')
        st.text('Configure plot params')
        form = st.selectbox('form', ['bar', 'step'])
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['form'] = form
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == InstantFreqAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop()
        min_max_width()
        st.text('')
        st.text('Configure plot params')
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == PoincareMapAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop()
        min_max_width()
        st.text('')
        st.text('Configure plot params')
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
        s = st.number_input('point size', 0.5, 20., 1.5, 0.5)
        exp_opt['experiment']['params_plot']['s'] = s
    elif exp_opt['experiment']['type'] == AutocorrelogramAnalyzer.ana_type:
        neuron_targets()
        st.text('')
        st.text('Configure data process params:')
        t_start_stop_bs()
        smooth_kernel()
        bias_start_stop()
        st.text('')
        st.text('Configure plot params')
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
        form = st.selectbox('form', ['bar', 'step'])
        exp_opt['experiment']['params_plot']['form'] = form
    elif exp_opt['experiment']['type'] == TuningAnalyzer.ana_type:
        neuron_targets()
        ts = st.columns(3)
        with ts[0]:
            delay = st.number_input("time delay(s)", 0., 10., 0., 0.001, format='%.4f')
        with ts[1]:
            n_dir = st.number_input("direction range count", 4, 128, 8, 1)
        with ts[2]:
            bin_size = st.number_input("bin_size", 0., 1., 0.5, 0.001, format='%.4f')
        smooth_kernel()
        kinematics = st.selectbox("kinematics column", behaviors)
        exp_opt['experiment']['kinematics'] = kinematics
        exp_opt['experiment']['params_data']['delay'] = delay
        exp_opt['experiment']['params_data']['n_dir'] = n_dir
        exp_opt['experiment']['params_data']['bin_size'] = bin_size
        st.text('')
        st.text('Configure plot params')
        plt_col = st.columns(2)
        with plt_col[0]:
            color = st.color_picker('color')
        with plt_col[1]:
            scatter = st.toggle("Scatter aligned")
        exp_opt['experiment']['params_plot']['color'] = color
        exp_opt['experiment']['params_plot']['scatter'] = scatter
    elif exp_opt['experiment']['type'] == PeriStimulusAnalyzer.ana_type:
        neuron_targets()
        st.divider()
        st.text('Configure data process params:')
        t_start_stop_bs()
        smooth_kernel()
        bias_start_stop()
        st.divider()
        st.text('Configure event params:')
        event_picker()
        st.divider()
        st.text("Configure plot params: ")
        plt_col = st.columns(2)
        with plt_col[0]:
            color = st.color_picker('color')
        with plt_col[1]:
            raster = st.toggle("Raster aligned", True)
        exp_opt['experiment']['params_plot']['color'] = color
        exp_opt['experiment']['params_plot']['raster_aligned'] = raster
    elif exp_opt['experiment']['type'] == CrossCorrelationAnalyzer.ana_type:
        st.text("Select a series of target neuron")
        neuron_targets()
        st.divider()
        st.text("Select a series of refer neuron")
        neuron_targets(True)
        st.divider()
        t_start_stop_bs()
        bias_start_stop()
        smooth_kernel()
        shift_p()
        st.divider()
        st.text('Configure event params:')
        event_picker()
        st.divider()
        st.text("Configure plot params: ")
        color = st.color_picker('color')
        exp_opt['experiment']['params_plot']['color'] = color
    elif exp_opt['experiment']['type'] == JointPSTHAnalyzer.ana_type:
        st.text("Select a series of target neuron")
        neuron_targets()
        st.divider()
        st.text("Select a series of refer neuron")
        neuron_targets(True)
        st.divider()
        t_start_stop_bs()
        bias_start_stop()
        smooth_kernel()
        shift_p()
        st.divider()
        st.text('Configure event params:')
        event_picker()


ce_configure_params()
# 参数预览
st.subheader("Experiment params preview")
st.json(exp_opt)

# 配置实验
exp_path = os.path.join('../experiments', exp_name)
exp_yml_path = os.path.join(exp_path, 'config.yml')
if st.button(f"Build experiment `{exp_name}`"):
    if exp_name == '' or exp_name is None:
        st.error("Please input a valid experiment name.")
    elif not os.path.exists(exp_path):
        os.makedirs(exp_path)
        dump_yml(exp_yml_path, exp_opt)
        st.success("Build success!")
    else:
        st.warning(f"{exp_name} already exist, if continue, original data will be covered!")

# 缓存配置
cache['last_experiment'] = exp_name
cache_dump(cache)
