from ndbox.utils.logger import get_root_logger
from .sua import TimeHistAnalyzer, CumActivityAnalyzer, RasterAnalyzer, ISIAnalyzer, ISITimeHistAnalyzer
from .sua import InstantFreqAnalyzer, AutocorrelogramAnalyzer, PoincareMapAnalyzer
from .mua import RasterMulAnalyzer
from .era import TuningAnalyzer
from .ana_base import Analyzer
from ndbox.utils import ANALYZER_REGISTRY
from ndbox.dataset import NWBDataset
import os

def ana_name_gen(nid: int, opt: dict, prev='Neuron', join_sep='.', precision=5, post='.png'):
    ana_name = [prev, str(nid)]
    for k, v in opt.items():
        if v is not None:
            ana_name.append(str(k))
            ana_name.append(str(v)[:precision])
    return join_sep.join(ana_name)+post

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

    experiment = kwargs.pop('experiment')
    target = experiment.pop('target')
    params_data = experiment.pop('params_data')
    params_plot = experiment.pop('params_plot')
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

    experiment = kwargs.pop('experiment')
    target = experiment.pop('target')
    params_data = experiment.pop('params_data')
    params_plot = experiment.pop('params_plot')
    spike_train = nwb_data.spike_train
    kin_resolution = nwb_data.bin_size
    kinematics = nwb_data.make_data(experiment["kinematics"]).values

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



