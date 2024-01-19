import os
import pandas as pd
import streamlit as st

from ndbox.utils import files_form_folder
from user_define_modules.result_plot import plot


@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath, index_col=0, header=0)


def round_list(data):
    return [round(num, 6) for num in eval(data)]


def report_ui():
    dir_path = os.path.join(root, 'results')
    results_dir = os.listdir(dir_path)
    results_dir = [r for r in results_dir if 'result' in r]

    page_name = st.sidebar.selectbox('Select Directory', results_dir + ['-'])
    if page_name != '-':
        result_dir = os.path.join(dir_path, page_name)
        if not os.path.exists(os.path.join(result_dir, 'hist.png')):
            plot(result_dir, root)
        st.image(os.path.join(result_dir, 'hist.png'))
        exp_folders = [entry for entry in os.listdir(result_dir)
                       if os.path.isdir(os.path.join(result_dir, entry))]

        for folder in exp_folders:
            st.subheader(str(folder))
            exp_path = os.path.join(result_dir, folder)
            metric_filenames = files_form_folder(exp_path, '*_metrics.csv')
            for metric_filename in metric_filenames:
                st.text(str(os.path.basename(metric_filename)))
                df = load_data(metric_filename)
                df = df.map(round_list)
                df.index.name = 'Model Name'
                st.dataframe(df)
            st.image(os.path.join(exp_path, 'trace.png'))
            st.image(os.path.join(exp_path, 'trace_compare.png'))
            st.image(os.path.join(exp_path, 'trace_3d.gif'))


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(__file__, os.path.pardir))
    report_ui()
