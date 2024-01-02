import streamlit as st
from st_pages import add_page_title
from app import cache_load, dump_yml, cache_dump
import os

add_page_title(layout="wide")

# 读取实验配置
cache = cache_load()
last_experiment = cache.get('last_experiment', None)

def run_exp():
    st.subheader(f"Execute experiment `{last_experiment}`")
    st.text("If already executed, it won't execute again and just read results.")
    # 读取结果文件
    exp_path = os.path.join("../experiments", last_experiment)
    result_dir = os.path.join(exp_path, 'result')
    clk = st.button(f"Execute `{last_experiment}`")
    if clk:
        if not os.path.exists(result_dir):
            os.system('python ../run_exp.py -e ' + last_experiment)
        show(result_dir)

def show(result_dir):
    image_names = os.listdir(result_dir)
    dir_list = [os.path.join(result_dir, im) for im in image_names]
    im_col2 = st.columns(2)
    left_half = int(len(dir_list) / 2)
    with im_col2[0]:
        for ii in range(left_half):
            st.image(dir_list[ii], caption=image_names[ii])
    with im_col2[1]:
        for ii in range(left_half, len(dir_list)):
            st.image(dir_list[ii], caption=image_names[ii])

if last_experiment is not None and last_experiment != '':
    run_exp()
else:
    st.error("You haven't create an experiment yet.")

