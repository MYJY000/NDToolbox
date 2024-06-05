"""
Cell Assembly Detection
    https://elephant.readthedocs.io/en/latest/reference/cell_assembly_detection.html
    `cell_assembly_detection`: CAD consists of a statistical parametric testing done on the level of 
    pairs of neurons, followed by an agglomerative recursive algorithm, in order to detect and test 
    statistically precise repetitions of spikes in the data. 

The Unitary Events Analysis
    https://elephant.readthedocs.io/en/latest/tutorials/unitary_event_analysis.html
    The Unitary Events (UE) analysis [1] tool allows us to reliably detect correlated spiking activity 
    that is not explained by the firing rates of the neurons alone. It was designed to detect coordinated 
    spiking activity that occurs significantly more often than predicted by the firing rates of the neurons. 
    The method allows one to analyze correlations not only between pairs of neurons but also between multiple 
    neurons, by considering the various spike patterns across the neurons. In addition, the method allows 
    one to extract the dynamics of correlation between the neurons by perform-ing the analysis in a 
    time-resolved manner. This enables us to relate the occurrence of spike synchrony to behavior.

Analysis of Sequences of Synchronous EvenTs
    https://elephant.readthedocs.io/en/latest/reference/asset.html
    Given two sequences of synchronous events (SSEs) sse1 and sse2, each consisting of a pool of 
    positions (iK, jK) of matrix entries and associated synchronous events SK, finds the intersection among them.

    
Spike Pattern Detection and Evaluation
    https://elephant.readthedocs.io/en/latest/reference/spade.html
    SPADE (Torre et al., 2013, Quaglio et al., 2017, Stella et al., 2019) is the combination of a 
    mining technique and multiple statistical tests to detect and assess the statistical significance 
    of repeated occurrences of spike sequences (spatio-temporal patterns, STP).

Cumulant Based Inference of higher-order Correlation (CuBIC)
    https://elephant.readthedocs.io/en/latest/reference/cubic.html
    CuBIC is a statistical method for the detection of higher order of correlations in parallel spike trains 
    based on the analysis of the cumulants of the population count.

Gaussian-Process Factor Analysis (GPFA)
    https://elephant.readthedocs.io/en/latest/reference/gpfa.html
    Gaussian-process factor analysis (GPFA) is a dimensionality reduction method (Yu et al., 2008) for neural 
    trajectory visualization of parallel spike trains. GPFA applies factor analysis (FA) to time-binned spike 
    count data to reduce the dimensionality and at the same time smoothes the resulting low-dimensional 
    trajectories by fitting a Gaussian process (GP) model to them.
"""
