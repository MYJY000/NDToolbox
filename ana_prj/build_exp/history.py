import os
import streamlit as st
from execute import show
from app import load_yml

exp_root = "../experiments"
experiments = os.listdir(exp_root)

# 1. 实验列表
st.subheader("Select an experiment to review")
exp_name = st.selectbox("History experiment list", experiments)
result_path = os.path.join(exp_root, exp_name, 'result')
config_path = os.path.join(exp_root, exp_name, 'config.yml')
log_path = os.path.join(exp_root, exp_name, 'log.log')
# 2. 实验结果
st.subheader("Experiment result")
show(result_path)
# 3. 实验配置
st.subheader(f"The configuration of experiment {exp_name}")
st.json(load_yml(config_path))
# 4. 实验日志
st.subheader(f"The experiment {exp_name} executing log")
with open(log_path, 'r', encoding='utf-8') as lg:
    txt = lg.read()
st.code(txt, language='log')
