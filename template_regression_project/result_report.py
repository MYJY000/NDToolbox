import os
import pandas as pd
import streamlit as st

from ndbox.utils import files_form_folder


@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath, index_col=0, header=0)


def round_list(data):
    return [round(num, 6) for num in eval(data)]


def report_ui():
    dir_path = os.path.join(root, 'results')
    results_dir = os.listdir(dir_path)

    page_name = st.sidebar.selectbox('Select Directory', results_dir + ['-'])
    if page_name != '-':
        result_dir = os.path.join(dir_path, page_name)
        exp_folders = [entry for entry in os.listdir(result_dir)
                       if os.path.isdir(os.path.join(result_dir, entry))]

        for folder in exp_folders:
            st.subheader(str(folder))
            metric_filenames = files_form_folder(os.path.join(result_dir, folder), '*_metrics.csv')
            for metric_filename in metric_filenames:
                st.text(str(os.path.basename(metric_filename)))
                df = load_data(metric_filename)
                df = df.map(round_list)
                df.index.name = 'Model Name'
                st.dataframe(df)


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(__file__, os.path.pardir))
    report_ui()
