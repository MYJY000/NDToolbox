from ndbox.utils.logger import get_root_logger
from .singleunit import TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer, ISIAnalyzer, ISITimeHistAnalyzer
from .singleunit import InstantFreqAnalyzer, AutocorrelogramAnalyzer, PoincareMapAnalyzer
from .correlation import RasterMulAnalyzer
from .era import TuningAnalyzer, PeriStimulusAnalyzer
from .rta import CrossCorrelationAnalyzer, JointPSTHAnalyzer
from .basic import Analyzer
from ndbox.utils import ANALYZER_REGISTRY
from ndbox.dataset import NWBDataset
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def ana_name_gen(nid: int, opt: dict, prev='Neuron', join_sep='.', precision=5, post='.png'):
    ana_name = [prev, str(nid)]
    for k, v in opt.items():
        if v is not None:
            ana_name.append(str(k))
            ana_name.append(str(v)[:precision])
    return join_sep.join(ana_name) + post


def sua_analysis(nwb_data: NWBDataset, SUAAnalyzer, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {SUAAnalyzer.name}")
    logger.info(SUAAnalyzer.description)

    target = kwargs.pop('target')
    params_data = kwargs.pop('params_data')
    params_plot = kwargs.pop('params_plot')
    spike_train = nwb_data.spike_train
    for t in target:
        analyzer: Analyzer = SUAAnalyzer(spike_train[t])
        analyzer.process(**params_data)
        ax = analyzer.plot(**params_plot)
        if ax is not None:
            res_name = ana_name_gen(t, analyzer.get_params_data())
            ax.figure.savefig(os.path.join(exp_result_path, res_name))
    logger.info(f"End the analysis of {SUAAnalyzer.name}")
    logger.info(f"All the results in {exp_result_path}")


@ANALYZER_REGISTRY.register()
def plot_time_hist(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, TimeHistAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_cum_activity(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, CumActivityAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_raster(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, RasterAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_isi_distribution(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, ISIAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_isi_time_hist(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, ISITimeHistAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_instant_freq(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, InstantFreqAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_autocorrelogram(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, AutocorrelogramAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_poincare_map(nwb_data: NWBDataset, **kwargs):
    sua_analysis(nwb_data, PoincareMapAnalyzer, **kwargs)


@ANALYZER_REGISTRY.register()
def plot_raster_mul(nwb_data: NWBDataset, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {RasterMulAnalyzer.name}")
    logger.info(RasterMulAnalyzer.description)

    experiment = kwargs.pop('experiment')
    target = experiment.pop('target')
    params_data = experiment.pop('params_data')
    params_plot = experiment.pop('params_plot')
    spike_train = nwb_data.spike_train
    spike_train = [spike_train[t] for t in target]

    analyzer = RasterMulAnalyzer(spike_train)
    analyzer.process(**params_data)
    ax = analyzer.plot(**params_plot)
    if ax is not None:
        res_name = ana_name_gen(target, analyzer.get_params_data())
        ax.figure.savefig(os.path.join(exp_result_path, res_name))
    logger.info(f"End the analysis of {RasterMulAnalyzer.name}")
    logger.info(f"Result saved in {exp_result_path}")


@ANALYZER_REGISTRY.register()
def plot_tuning_curve(nwb_data: NWBDataset, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {TuningAnalyzer.name}")
    logger.info(TuningAnalyzer.description)

    target = kwargs.pop('target')
    params_data = kwargs.pop('params_data')
    params_plot = kwargs.pop('params_plot')
    spike_train = nwb_data.spike_train
    kin_resolution = nwb_data.bin_size
    kinematics = nwb_data.make_data(kwargs["kinematics"]).values

    for t in target:
        analyzer = TuningAnalyzer(spike_train[t], kinematics, kin_resolution,
                                  logger_name=logger_name)
        analyzer.process(**params_data)
        ax = analyzer.plot(**params_plot)
        if ax is not None:
            res_name = ana_name_gen(t, analyzer.get_params_data())
            ax.figure.savefig(os.path.join(exp_result_path, res_name))
        logger.info(f"End the analysis of {RasterMulAnalyzer.name}")
        logger.info(f"Result saved in {exp_result_path}")


@ANALYZER_REGISTRY.register()
def plot_peri_stimulus_time_histogram(nwb_data: NWBDataset, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {PeriStimulusAnalyzer.name}")
    logger.info(PeriStimulusAnalyzer.description)

    target = kwargs.pop('target')
    events = kwargs.pop('events')
    params_data = kwargs.pop('params_data')
    params_plot = kwargs.pop('params_plot')
    spike_train = nwb_data.spike_train
    raster_aligned = params_plot.pop('raster_aligned')

    for t in target:
        analyzer = PeriStimulusAnalyzer(spike_train[t], np.array(events), logger_name=logger_name)
        analyzer.process(**params_data)
        if raster_aligned:
            fig, ax = plt.subplots(2, 1)
            analyzer.raster(ax[0], **params_plot)
            analyzer.plot(axes=ax[1], **params_plot)
        else:
            fig, ax = plt.subplots()
            analyzer.plot(axes=ax, **params_plot)
        res_name = ana_name_gen(t, analyzer.get_params_data())
        fig.savefig(os.path.join(exp_result_path, res_name))
        logger.info(f"End the analysis of {RasterMulAnalyzer.name}")
        logger.info(f"Result saved in {exp_result_path}")


@ANALYZER_REGISTRY.register()
def plot_cross_correlogram(nwb_data: NWBDataset, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {CrossCorrelationAnalyzer.name}")
    logger.info(PeriStimulusAnalyzer.description)

    target = kwargs.pop('target')
    events = kwargs.pop('events')
    refer = kwargs.pop('refer')
    params_data = kwargs.pop('params_data')
    params_plot = kwargs.pop('params_plot')
    spike_train = nwb_data.spike_train

    for t in target:
        for r in refer:
            analyzer = CrossCorrelationAnalyzer(spike_train[t], spike_train[r],
                                                event_train=np.array(events), logger_name=logger_name)
            analyzer.process(**params_data)
            ax = analyzer.plot(**params_plot)
            para = analyzer.get_params_data()
            para['smooth'] = None
            para['kernel_size'] = None
            para['t_start'] = None
            para['t_stop'] = None
            para['refer'] = r
            res_name = ana_name_gen(t, para)
            ax.figure.savefig(os.path.join(exp_result_path, res_name))
            logger.info(f'cross analysis of {t} and {r} success')
    logger.info(f"End the analysis of {RasterMulAnalyzer.name}")
    logger.info(f"Result saved in {exp_result_path}")


@ANALYZER_REGISTRY.register()
def plot_joint_psth(nwb_data: NWBDataset, **kwargs):
    exp_root_path = kwargs.pop('root')
    exp_result_path = os.path.join(exp_root_path, 'result')
    logger_name = kwargs.pop('logger_name')
    logger = get_root_logger(logger_name=logger_name)
    if os.path.exists(exp_result_path):
        logger.warning(f'{exp_result_path} already exist')
    else:
        os.makedirs(exp_result_path)
    logger.info(f"Start the analysis of {JointPSTHAnalyzer.name}")
    logger.info(JointPSTHAnalyzer.description)

    target = kwargs.pop('target')
    events = kwargs.pop('events')
    refer = kwargs.pop('refer')
    params_data = kwargs.pop('params_data')
    params_plot = kwargs.pop('params_plot')
    spike_train = nwb_data.spike_train

    for t in target:
        for r in refer:
            analyzer = JointPSTHAnalyzer(spike_train[t], spike_train[r],
                                         event_train=np.array(events), logger_name=logger_name)
            analyzer.process(**params_data)
            ax = analyzer.plot(**params_plot)
            plt.close()
            para = analyzer.get_params_data()
            para['smooth'] = None
            para['kernel_size'] = None
            para['t_start'] = None
            para['t_stop'] = None
            para['refer'] = r
            res_name = ana_name_gen(t, para)
            ax.figure.savefig(os.path.join(exp_result_path, res_name))
            logger.info(f'j-psth analysis of {t} and {r} success')
    logger.info(f"End the analysis of {RasterMulAnalyzer.name}")
    logger.info(f"Result saved in {exp_result_path}")
