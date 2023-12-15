from copy import deepcopy
from .ana_base import Analyzer, Neuron, NeuronList
from .analysis import *
from .mua import *
from .sua import *
from .rta import *
pipeline_path = 'pipeline'
pipeline_ui_path = 'pipeline_ui'

def run_analyze(nwb_data, opt):
    """
    Build analyze from options.

    :param nwb_data: NWBDataset.
    :param opt: dict. Configuration. It must contain:
        type - str. Metric type.
    """

    opt = deepcopy(opt)
    experiment = opt.get('experiment')
    analyze_type = experiment.get('type')
    result = ANALYZER_REGISTRY.get(analyze_type)(nwb_data, **opt)
    return result
