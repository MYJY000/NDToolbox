import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
from collections import OrderedDict
from os import path

from ndbox.model import build_model
from ndbox.utils import yaml_load, load_image


def trace_plot(index, y_true, y_pred, save_path, model_name):
    start = 0
    end = 100
    index = index[start:end]
    y_true = y_true[start:end]
    y_pred = y_pred[start:end]

    t = list(index / np.timedelta64(1, 's'))
    x1 = y_true[:, 0]
    y1 = y_true[:, 1]
    p1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
    seg1 = np.concatenate([p1[:-1], p1[1:]], axis=1)
    x2 = y_pred[:, 0]
    y2 = y_pred[:, 1]
    p2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
    seg2 = np.concatenate([p2[:-1], p2[1:]], axis=1)

    lc1 = LineCollection(seg1, cmap='viridis', linewidths=2, linestyles='-', norm=plt.Normalize(0, t[-1]))
    lc1.set_array(t)
    lc2 = LineCollection(seg2, cmap='viridis', linewidths=2, linestyles='-', norm=plt.Normalize(0, t[-1]))
    lc2.set_array(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    lines1 = ax1.add_collection(lc1)
    lines2 = ax2.add_collection(lc2)
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-20, 130)
    ax2.set_xlim(-100, 100)
    ax1.set_title('True Finger Position')
    ax2.set_title(f'{model_name}')
    plt.colorbar(lines2, label='Time (s)')
    plt.savefig(path.join(save_path, 'trace.png'), dpi=300, bbox_inches='tight')


def hist_plot(label, r2_val, cc_val, result_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.bar(label, r2_val, width=0.5)
    ax2.bar(label, cc_val, width=0.5)
    ax1.set_title('R2')
    ax1.set_xticklabels(label, rotation=90, fontsize=8)
    ax2.set_title('CC')
    ax2.set_xticklabels(label, rotation=90, fontsize=8)
    plt.savefig(path.join(result_dir, 'hist.png'), dpi=300, bbox_inches='tight')


def plot(result_dir, root):
    temp_path = path.join(root, 'temp_files')
    config_path = path.join(result_dir, 'config.yml')
    opt = yaml_load(config_path)
    datasets_opt = opt.get('dataset', OrderedDict())
    experiment_opt = opt.get('experiment', OrderedDict())

    exp_folder = [entry for entry in os.listdir(result_dir)
                  if path.isdir(path.join(result_dir, entry))]

    model_label = []
    r2_val = []
    cc_val = []
    for folder in exp_folder:
        exp_name = str(folder)
        exp_opt = experiment_opt.get(exp_name)
        exp_path = path.join(result_dir, exp_name)
        if exp_opt is not None:
            processor_opt = exp_opt.get('processor', OrderedDict())
            model_opt = exp_opt.get('model', OrderedDict())
            train_opt = exp_opt.get('train', OrderedDict())

            # load model
            model_load_path = path.join(exp_path, 'model')
            _mode_path = os.listdir(model_load_path)
            model_name = _mode_path[0].split('.')[0]
            model_label.append(model_name)
            model_load_path = path.join(model_load_path, _mode_path[0])
            model = build_model(model_opt)
            model.load(model_load_path)

            # save output
            train_dataset_name = train_opt.get('dataset')
            if train_dataset_name is not None:
                dataset = load_image(
                    temp_path,
                    datasets_opt[train_dataset_name],
                    processor_opt.get(train_dataset_name)
                )
                target = train_opt['target']
                _, x = dataset.get_spike_array()
                _, y_true = dataset.get_behavior_array(target)
                index = dataset.data.index
                y_pred = model.predict(x)
                trace_plot(index, y_true, y_pred, exp_path, model_name)

        metric_filename = path.join(exp_path, 'train_metrics.csv')
        df = pd.read_csv(metric_filename, index_col=0, header=0)
        df = df.map(lambda u: np.array(eval(u)))
        row_mean = df.iloc[0].mean()
        r2_val.append(row_mean[0])
        cc_val.append(row_mean[1])
    hist_plot(model_label, r2_val, cc_val, result_dir)
