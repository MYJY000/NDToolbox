import streamlit as st
from st_pages import add_page_title
from app import cache_load, cache_dump
import pandas as pd
import sys
sys.path.append("../../")
from ndbox.dataset import NWBDataset

add_page_title(layout="wide")
st.header("Choose a dataset below")
# 供选择的数据库
dataset_repo = {
    "sub-Indy": r"D:\datasets\NWBDemo\sub-Indy_desc-train_behavior+ecephys.nwb",
    "sub-Han": r"D:\datasets\NWBDemo\sub-Han_desc-train_behavior+ecephys.nwb",
    "sub-Haydn": r"D:\datasets\NWBDemo\sub-Haydn_desc-train_ecephys.nwb",
    "sub-Jenkins": r"D:\datasets\NWBDemo\sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb",
    "sub-210862": r"D:\datasets\NWBDemo\sub-210862_ses-20130626_behavior+ecephys+ogen.nwb",
    "sub-219030": r"D:\datasets\NWBDemo\sub-219030_ses-20130901_behavior+ecephys+ogen.nwb",
    "sub-BH454-A": r"D:\datasets\NWBDemo\sub-BH454_obj-3sutmh_ecephys.nwb",
    "sub-BH454-B": r"D:\datasets\NWBDemo\sub-BH454_obj-78d8l9_ecephys.nwb",
}

# 缓存数据结构
cache: dict = cache_load()

# 选择数据集, 注意缓存当次选择
last_dataset = cache.get('last_dataset', None)
if last_dataset is None or last_dataset == '-':
    dataset = st.selectbox('select a dataset', ['-']+list(dataset_repo.keys()))
else:
    dlt = list(dataset_repo.keys())
    idx = dlt.index(last_dataset)+1
    dataset = st.selectbox('select a dataset', ['-'] + list(dataset_repo.keys()), index=idx)
cache['last_dataset'] = dataset
cache_dump(cache)
# 加载数据集基本信息
verbose_a = None
verbose_b = None
if dataset != '-' and dataset not in cache:
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
    behaviors = list(nwb_data.data_dict.keys())
    events = {}
    if 'trials' in behaviors:
        import numpy as np
        trials = nwb_data.make_data("trials")
        start_time = trials['start_time'].values
        stop_time = trials['stop_time'].values
        stamp = list(np.around((start_time+stop_time)/2, 3))
        stamp = [float(sta) for sta in stamp]
        events["event_local"] = stamp
    current_dataset_info['bin_size'] = nwb_data.bin_size
    current_dataset_info['duration'] = nwb_data.data.shape[0] * nwb_data.bin_size
    current_dataset_info['neuron_count'] = nwb_data.spike_train.size
    firing_rates = [round(sp.size / current_dataset_info['duration'], 2) for sp in nwb_data.spike_train]
    current_dataset_info['min avg rates'] = min(firing_rates)
    current_dataset_info['max avg rates'] = max(firing_rates)

    cache[dataset] = {
        'info': current_dataset_info,
        'behaviors': behaviors,
        'events': events,
        'firing_rates': firing_rates,
        'verbose_a': nwb_data.content_repr(nwb_data.content_dict),
        'verbose_b': nwb_data.data_info(),
    }
    cache_dump(cache)

# [界面]: 展示数据集信息
if cache is not None and dataset in cache:
    st.subheader(f"Basic Information about dataset `{dataset}`")
    current_dataset_info = cache[dataset]['info']
    bi_df = pd.DataFrame(index=list(current_dataset_info.keys()), columns=['values'],
                         data=list(current_dataset_info.values()))
    st.dataframe(bi_df)

if cache is not None and dataset in cache:
    st.subheader(f"Detailed Information about dataset `{dataset}`")
    st.text(cache[dataset]['verbose_a'])
    st.text(cache[dataset]['verbose_b'])


