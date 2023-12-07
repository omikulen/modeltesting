"""
## Overview

The module provides utility functions for distinguishing between different models in particle physics experiments. The module assumes that observations consist of \( N \) channels, each parameterized by effective branching ratios \( \text{br}_i \) and backgrounds \( b_i \).

The theoretical predictions for the number of events in each channel are given by:

\[
N_i = N \times \text{br}_i + b_i
\]

where \( N \) is the total number of events.

Given a real model \( \text{br1}_i \), the utility functions aim to answer how many events \( N \) are required to exclude a tested model \( \text{br2}_i \). Specifically, the module calculates at what \( N \) the probability to exclude the tested model becomes significant.

"""


import numpy as np
import warnings





def find_bestfit_N(data:np.ndarray, 
                   branchings:np.ndarray, 
                   backgrounds:np.ndarray):
    """
    Finds the best-fit number of events (N_events) for each sample in the data, as well as the corresponding chi-squared value.
    
    Parameters:
    -----------
    data : array-like
        An array of shape (Nsamples, Nchannels) representing observed data for each channel and sample.
        
    branchings : array-like
        An array of shape (Nchannels,) representing the branching ratios for each channel.
        
    backgrounds : array-like
        An array of shape (Nchannels,) representing the background events for each channel.
        
    Returns:
    --------
    tuple (Nev, chi2)
        Nev : np.ndarray
            An array of shape (Nsamples,) representing the best-fit N_events for each sample in the data. Infinities for samples that cannot be fit.
            
        chi2 : np.ndarray
            An array of shape (Nsamples,) representing the chi-squared value for the best-fit N_events for each sample. Infinities for samples that cannot be fit.
            
    Example:
    --------
    >>> find_bestfit_N(np.array([[10, 20], [15, 25]]), np.array([0.4, 0.6]), np.array([1, 1]))
    (array([25.5, 33.75]), array([0.42, 0.37]))
    """


    data, branchings, backgrounds = np.array(data, dtype = float), np.array(branchings), np.array(backgrounds)
    
    if len(data.shape) != 2:
        data = data[:, np.newaxis]

    if data.shape[1] != branchings.shape[0] or data.shape[1] != backgrounds.shape[0] or branchings.shape[0] != backgrounds.shape[0]:
        raise ValueError(f'The number of channels in data ({data.shape[1]}), branchings ({branchings.shape[0]}), and backgrounds ({backgrounds.shape[0]}) must be the same.)')

    if np.any(branchings < 0) or np.any(backgrounds < 0):
        raise ValueError(f'Branchings and backgrounds must be non-negative.')

    if np.any(data < 0):
        warnings.warn(f'There are negative counts in data.')	

    Nsamples = data.shape[0]
    resulting_Nev = np.zeros(Nsamples, dtype = np.float64)


    signal = data - backgrounds 

    #separate into backgrounds and signal dominated channels
    mask_bkg = signal < backgrounds # 2d bool array of shape (Nsamples, Nchannels)
    mask_signal = signal >= backgrounds 

    # drop signal-dominated realizations where branching = 0 while signal != 0 for any channel. These are excluded immediately
    drop_indices = np.any((mask_signal) & (signal != 0) & (branchings == 0), axis = 1) # 1d bool array of shape (Nsamples,)

    # otherwise ignore channels with branching = 0
    skipped_bins = (mask_bkg) & (branchings == 0)

    resulting_Nev[drop_indices] = np.inf

    # take the remaining realizations
    good_indices = ~drop_indices # 1d bool array of shape (Nsamples,)
    mask_good_bkg = mask_bkg & good_indices[:, np.newaxis] # 2d bool array of shape (Nsamples, Nchannels)
    mask_good_signal = mask_signal & good_indices[:, np.newaxis] 


    branchings_2d = np.tile(branchings, (Nsamples, 1))
    backgrounds_2d = np.tile(backgrounds, (Nsamples, 1))

    _ = np.zeros_like(data)
    _[mask_good_bkg] = (branchings_2d[mask_good_bkg])**2/(backgrounds_2d[mask_good_bkg]+1e-10) # filling in 2D array
    C1 = 2*np.sum(_, axis = 1) # sum over channels into 1D array

    _ = np.zeros_like(data)
    _[mask_good_signal] = branchings_2d[mask_good_signal]
    C2 = np.sum(_, axis = 1)

    _ = np.zeros_like(data)
    _[mask_good_bkg] = (signal[mask_good_bkg]*branchings_2d[mask_good_bkg])/(backgrounds_2d[mask_good_bkg] +1e-10)
    C3 = 2*np.sum(_, axis = 1)

    _ = np.zeros_like(data)
    _[mask_good_signal] = (signal[mask_good_signal])**2/(branchings_2d[mask_good_signal]+1e-10)
    C4 = np.sum(_, axis = 1)

    
    # If backgrounds dominates all channels (ignoring skipped bins)
    type1_condition = np.all(mask_good_bkg | (skipped_bins), axis = 1)
    type1_indices = good_indices & type1_condition

    N_type1 = C3[type1_condition]/C1[type1_condition]
    resulting_Nev[type1_indices] = N_type1

    # if signal dominates all channels
    type2_condition = np.all(mask_good_signal | (skipped_bins), axis = 1)
    type2_indices = good_indices & type2_condition

    N_type2 = np.sqrt(C4[type2_condition]/C2[type2_condition])
    resulting_Nev[type2_indices] = N_type2


    # general case: some channels are bkg-dominated, some are signal-dominated
    type3_condition = (good_indices) & (~type1_condition) & (~type2_condition)
    type3_indices = good_indices & type3_condition
    

    a = (C2[type3_condition] - C3[type3_condition])/(3*C1[type3_condition])
    b = C4[type3_condition]/C1[type3_condition]
    N_type3 = np.zeros_like(a)

    # handle real and complex discriminants separately
    real_discriminant = b>=4*a**3
    N_type3[real_discriminant] = (np.sqrt(b[real_discriminant]) + np.sqrt(b[real_discriminant] - 4*a[real_discriminant]**3))**2/4
    N_type3[real_discriminant] = N_type3[real_discriminant]**(1/3) 
    N_type3[real_discriminant] += a[real_discriminant]**2/N_type3[real_discriminant]
    N_type3[real_discriminant] -= a[real_discriminant]

    complex_discriminant = b<4*a**3
    phi = 2/3*np.arctan(np.sqrt(4*a[complex_discriminant]**3/b[complex_discriminant] - 1))
    N_type3[complex_discriminant] = (2*np.cos(phi) - 1)*a[complex_discriminant]

    resulting_Nev[type3_indices] = N_type3

    ### remove negative solutions
    resulting_Nev[resulting_Nev < 0] = 0

    # calculate chi2
    resulting_chi2 = np.zeros_like(resulting_Nev)
    resulting_chi2[good_indices] = np.sum((signal[good_indices] - resulting_Nev[good_indices][:, np.newaxis]*branchings_2d[good_indices])**2/(resulting_Nev[good_indices][:, np.newaxis]*branchings_2d[good_indices] + backgrounds_2d[good_indices]+1e-10), axis = 1)
    resulting_chi2[~good_indices] = np.inf
    
    return resulting_Nev, resulting_chi2


    




def compare_two_models(N_real:float, 
                       branchings_real:np.ndarray, 
                       branchings_tested:np.ndarray, 
                       backgrounds = None, 
                       CL = 0.9, 
                       Nsamples = 100000):
    """
N_real - total expected number of events in the real model
branchings_real - branching ratios in the real model
branchings_tested - branching ratios in the tested model

backgrounds - backgrounds in each channel. If None, assume zero backgrounds
CL - confidence level for the exclusion
Nsamples - number of realizations to generate

Here we simulate data from the real model, estimated the best-fit N_bestfit, and compute chi2 for the tested model.

The probability to exclude the tested model is computed as the fraction of realizations with chi2 > chi2_threshold.
chi2_threshold is estimated for the model with N_tested = N_f_bestfit, which is the mean of the N_f distribution for good realizations with chi2 < 20.
Returns: (probability, N_f_bestfit)
    """

    if len(branchings_real) != len(branchings_tested):
        raise ValueError(f'Number of channels in real and tested models must be the same.')
    if backgrounds is None:
        backgrounds = np.zeros(len(branchings_real))
    elif len(backgrounds) != len(branchings_real):
        raise ValueError(f'Number of channels in the models and background must be the same.')
    
    branchings_real, branchings_tested, backgrounds = np.array(branchings_real), np.array(branchings_tested), np.array(backgrounds)

    if np.any(branchings_real < 0) or np.any(branchings_tested < 0) or np.any(backgrounds < 0) or N_real < 0:
        raise ValueError(f'Nreal, branchings and background must be non-negative.')
    if CL < 0 or CL > 1:
        raise ValueError(f'Confidence level must be between 0 and 1.')
    if Nsamples < 1 or not isinstance(Nsamples, int):
        raise ValueError(f'Nsamples must be a positive integer.')

    

    if np.any(branchings_real < 0) or np.any(branchings_tested < 0) or np.any(backgrounds < 0):
        raise ValueError(f'Branchings and background must be non-negative.')


    data = np.random.poisson(N_real*branchings_real + backgrounds, size = (Nsamples, len(branchings_real)))

    N_bestfit, chi2_bestfit = find_bestfit_N(data, branchings_tested, backgrounds)

    too_large_chi2 = chi2_bestfit > 20
    if np.all(too_large_chi2):
        return 1, N_real # if all chi2 are too large (including np.inf), exclusion probability 100% without any further checks 


    N_bestfit = N_bestfit[~too_large_chi2]
    chi2_bestfit = chi2_bestfit[~too_large_chi2]
    
    # we cannot compute CDF for each N_bestfit separately, it's too time consuming
    # so we assume that they all roughly equal and compute only it for the mean value of N_bestfit
    N_bestfit_mean = np.mean(N_bestfit)

    tested_predictions = N_bestfit_mean*branchings_tested + backgrounds + 1e-10
    data_to_compare = np.random.poisson(tested_predictions, size = (Nsamples, len(branchings_real)))

    chi2_to_compare = np.sum( (data_to_compare - tested_predictions)**2/tested_predictions, axis = 1)

    chi2_threshold = np.percentile(chi2_to_compare, CL*100) # chi2_threshold is the chi2 value with p-value 1-CL
    return (np.sum(too_large_chi2) + np.sum(chi2_bestfit > chi2_threshold))/Nsamples, N_bestfit_mean




def find_estimated_N_real(branchings_real, 
                         branchings_tested, 
                         backgrounds = None, 
                         CL = 0.9, 
                         df = None):
    """
    Computes a rough estimate of the required N_real and the corresponding best-fit N_tested.
    
    Parameters:
    -----------
    branchings_real : array-like
        Branching ratios in the real model.
        
    branchings_tested : array-like
        Branching ratios in the tested model.
        
    backgrounds : array-like, optional
        Backgrounds in each channel. If None (default), zero backgrounds are assumed.
        
    CL : float, optional
        Confidence level for the estimation. Default is 0.9.
        
    df : int or None, optional
        Degrees of freedom. If None (default), it is inferred from the data.
        
    Returns:
    --------
    tuple
        A tuple containing the estimated N_real and N_tested values.
        
    Raises:
    -------
    ValueError
        If the lengths of branchings_real, branchings_tested, and backgrounds do not match.
        
    Examples:
    ---------
    >>> find_estimated_N_real([0.4, 0.6], [0.1,  0.1], backgrounds=[50, 100])
    (94.96566842987566, 474.8283421493783)
    """

    
    from scipy.stats import chi2

    branchings_real = np.array(branchings_real, dtype = float)
    branchings_tested = np.array(branchings_tested, dtype = float)

    if np.any(branchings_real < 0) or np.any(branchings_tested < 0):
        raise ValueError(f'Branchings must be non-negative.')

    if len(branchings_real) != len(branchings_tested):
        raise ValueError(f'Number of channels in real and tested models must be the same.')
    
    if backgrounds is None:
        backgrounds = np.zeros(len(branchings_real))
    elif len(backgrounds) != len(branchings_real):
        raise ValueError(f'Number of channels in the models and background must be the same.')
    
    backgrounds = np.array(backgrounds, dtype = float)
    if np.any(branchings_real < 0) or np.any(branchings_tested < 0):
        raise ValueError(f'Branchings must be non-negative.')

    if df is None:
        df = len(branchings_real) - 1

    # background-free estimate
    l = np.sum(branchings_tested)/np.sum(branchings_real)

    denominator = np.sum((l*branchings_real - branchings_tested)**2/(branchings_tested+1e-10))
    N_real = l*chi2.isf(1 - CL, df = df)/denominator+1
    N_tested = N_real/l

    # background dominated estimate

    N_real_bkg = np.sqrt(chi2.isf(1 - CL, df = df)/np.sum((branchings_real - branchings_tested/l)**2/(backgrounds+1e-10)))
    N_tested_bkg = N_real_bkg/l

    

    return (N_real, N_tested) if N_real > N_real_bkg else (N_real_bkg, N_tested_bkg)




def _fromfile(path):
    """
    Reads a .txt, .csv file and extracts branching ratios, background, and efficiencies to construct an output dictionary.
    
    Parameters:
    -----------
    path : str
        Path to the file. The file should contain columns with the following headers:
        - 'br1': Branching ratios in the real model
        - 'br2': Branching ratios in the tested model
        - 'bkg': Background in each channel (optional, default is 0)
        - 'eff1': Efficiencies in each channel for the real model (optional, default is 1)
        - 'eff2': Efficiencies in each channel for the tested model (optional, default is 1)
        - 'channel_names': Channel labels (optional)
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'branchings_real': Calculated as br1 * eff1
        - 'branchings_tested': Calculated as br2 * eff2
        - 'backgrounds': Read from the 'bkg' column
        - 'channel_names': Names of the channels if provided
        
    Example File Format:
    --------------------
    br1,br2,bkg,eff1 
    0.4,0.3,50,0.9
    0.6,0.7,100.,0.95
    
    Examples:
    ---------
    >>> _fromfile('path/to/file.csv')
    {
        'branchings_real': [0.36, 0.57],
        'branchings_tested': [0.27, 0.665],
        'backgrounds': [50., 100.],
        'names': None
    }
    """
    
    if path.endswith('.csv') or path.endswith('.txt'):
        for sep in (",", ";", "\t", " "):
            try:
                data = np.loadtxt(path, dtype=str, delimiter=sep)
                if len(data.shape) != 2:
                    continue
                data[0] = np.array([i.lower().strip(" ") for i in data[0]])
                break
            except ValueError:
                continue
        else:
            raise ValueError("Unknown separator or malformed file")
    else:
        raise ValueError(f'Unknown file format {path}. Only .csv and .txt are supported.')


    headers = data[0]
    try:
        name_index = np.where(headers == 'channel_names')[0][0]
        names = data[1:, name_index]
        data = np.delete(data, name_index, axis = 1)
        headers = np.delete(headers, name_index)
    except IndexError:
        names = None
    
    data = data[1:].astype(float)

    #### handle cases
    try:
        branchings_real = data[:, np.where(headers == 'br1')[0][0]]
    except IndexError:
        raise ValueError(f'No column with header br1 found in {path}.')
    
    try:
        branchings_tested = data[:, np.where(headers == 'br2')[0][0]]
    except IndexError:
        raise ValueError(f'No column with header br2 found in {path}.')
    
    try:
        backgrounds = data[:, np.where(headers == 'bkg')[0][0]]
    except IndexError:
        backgrounds = None

    try:
        efficiencies = data[:, np.where(headers == 'eff')[0][0]]
    except IndexError:
        efficiencies = np.ones_like(branchings_real)




    branchings_real *= efficiencies
    branchings_tested *= efficiencies

    output ={
        'branchings_real': branchings_real,
        'branchings_tested': branchings_tested,
        'backgrounds': backgrounds,
        'channel_names': names
    }

    return output



def find_optimal_N_real(
    branchings_real = None, 
    branchings_tested = None,    
    backgrounds = None,
    path = None,
    exclusion_probability = 0.9, 
    CL = 0.9, 
    Nsamples = 100000, 
    convergence_rate = 1.):
    """
    Finds the optimal N_real value that provides the desired exclusion probability using a chi2 test.
    
    Parameters:
    -----------
    branchings_real : array-like, optional
        Branching ratios in the real model, calculated as br1 * eff.
        
    branchings_tested : array-like, optional
        Branching ratios in the tested model, calculated as br2 * eff.
        
    backgrounds : array-like, optional
        Backgrounds in each channel. If None (default), zero backgrounds are assumed.
        
    path : str, optional
        Path to a file containing columns for:
        - 'br1': Branching ratios in the real model
        - 'br2': Branching ratios in the tested model
        - 'bkg': Backgrounds in each channel (optional, default 0)
        - 'eff': Efficiencies in each channel (optional, default 1)
        
    exclusion_probability : float, optional
        Desired probability to exclude the tested model. Default is 0.9.
        
    CL : float, optional
        Confidence level for the model rejection with chi2 test. Default is 0.9.
        
    Nsamples : int, optional
        Number of realizations to use for the chi2 test. Default is 100000.
        
    convergence_rate : float, optional
        Rate of convergence for the gradient descent algorithm
            N_real = N_real*(1 + convergence_rate*(exclusion_probability - current_probability))
        Default is 1.
        
    Returns:
    --------
    tuple
        A tuple containing:
        - 'Nreal': The optimal N_real value
        - 'Ntested': Roughly the best-fit N events for the tested model corresponding to N_real
        - 'probability': The achieved exclusion probability
        
    Example:
    --------
    >>> find_optimal_N_real(branchings_real=[1.0, 0.5], branchings_tested=[0.5, 1.0])
    (16.671661862408424, 19.84366716420989, 0.8935)

    >>> find_optimal_N_real(branchings_real=[1.0, 0.5], branchings_tested=[0.5, 1.0], backgrounds=[100, 200])
    (65.44228924615766, 64.83965225789879, 0.9029)
    
    """
        
    if path is not None:
        output = _fromfile(path)
        branchings_real = output['branchings_real']
        branchings_tested = output['branchings_tested']
        backgrounds = output['backgrounds']
    else:
        branchings_real = np.array(branchings_real, dtype = float)
        branchings_tested = np.array(branchings_tested, dtype = float)
    if backgrounds is None:
        backgrounds = np.zeros(len(branchings_real))
    else:
        backgrounds = np.array(backgrounds, dtype = float)

    if len(branchings_real) != len(branchings_tested):
        raise ValueError(f'Number of channels in real and tested models must be the same.')
    elif len(backgrounds) != len(branchings_real):
        raise ValueError(f'Number of channels in the models and background must be the same.')
    if np.any(branchings_real < 0) or np.any(branchings_tested < 0):
        raise ValueError(f'Branchings must be non-negative.')


    # starting point of the scan
    current_Nreal, _ = find_estimated_N_real(branchings_real, branchings_tested, backgrounds = backgrounds, CL = CL)
    current_probability, current_Ntested = compare_two_models(
            current_Nreal, branchings_real, branchings_tested, backgrounds = backgrounds,
            CL = CL, Nsamples = Nsamples)

    best_probability, best_Nreal, best_Ntested = current_probability, current_Nreal, current_Ntested


    for isteps in range(50):
        # we are performing a gradient descent for (exclusion_probability - y)**2, but logarithmically
        current_Nreal *= (1 + convergence_rate*(exclusion_probability-current_probability))
        current_probability, current_Ntested = compare_two_models(
                current_Nreal, branchings_real, branchings_tested, backgrounds = backgrounds,
                CL = CL, Nsamples = Nsamples)

        if abs(current_probability - exclusion_probability) < abs(best_probability - exclusion_probability):
            best_probability = current_probability
            best_Nreal = current_Nreal
            best_Ntested = current_Ntested
        if abs(current_probability - exclusion_probability) < 0.01:
            # stop after reaching a percent accuracy
            break
    return best_Nreal, best_Ntested, best_probability


def barplot(ax, N_real, N_tested, 
    branchings_real = None, 
    branchings_tested = None, 
    path = None, **kwargs):

    """
    Creates a bar plot to visualize the results of the scan over N_real on a given axis.
    
    Parameters:
    -----------
    ax : matplotlib axis object
        The axis on which to plot the bar chart.

    N_real : float or int
        The N_real value.

    N_tested : float or int
        The N_tested value
    
    branchings_real : array-like, optional
        Branching ratios in the real model. Ignored if 'path' is provided.
        
    branchings_tested : array-like, optional
        Branching ratios in the tested model. Ignored if 'path' is provided.
        
    path : str, optional
        Path to a file containing branching ratios and background data. If provided, 'branchings_real' and 'branchings_tested' are ignored.
        
    **kwargs : dict, optional
        Additional keyword arguments for customization. Possible keys are:
        - 'channel_names': Names of the channels for labeling
        - 'backgrounds': Backgrounds in each channel

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> barplot(ax, 100, 90, branchings_real=[0.2, 0.4, 0.6], branchings_tested=[0.3, 0.6, 0.5], backgrounds = [10, 20, 30], names = ['First', 'Second', 'Third'])
    """
            
    channel_names = None

    if path is not None:
        output = _fromfile(path)
        branchings_real = output['branchings_real']
        branchings_tested = output['branchings_tested']
        backgrounds = output['backgrounds']
        if "names" in output:
            channel_names = output['names']
    else:
        branchings_real = np.array(branchings_real, dtype = float)
        branchings_tested = np.array(branchings_tested, dtype = float)
        if 'channel_names' in kwargs:
            channel_names = kwargs['channel_names']
        backgrounds = kwargs.get('backgrounds', None)
    if backgrounds is None:
        backgrounds = np.zeros_like(branchings_real)
    else:
        backgrounds = np.array(backgrounds, dtype = float)

    

    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(len(branchings_real))]

    # Loop over each pair of bars

    for i, (b1, b2, b3) in enumerate(zip(N_real*branchings_real, N_tested*branchings_tested, backgrounds)):
        # Plot the taller bar first
        if b1 > b2:
            ax.bar(channel_names[i], b1, color='blue', alpha=0.4, edgecolor='black', linewidth=1.5, label='_nolegend_', bottom=b3)
            ax.bar(channel_names[i], b2, color='red', alpha=0.4, edgecolor='black', linewidth=1.5, label='_nolegend_', bottom=b3)
        else:
            ax.bar(channel_names[i], b2, color='red', alpha=0.4, edgecolor='black', linewidth=1.5, label='_nolegend_', bottom=b3)
            ax.bar(channel_names[i], b1, color='blue', alpha=0.4, edgecolor='black', linewidth=1.5, label='_nolegend_', bottom=b3)
        ax.bar(channel_names[i], b3, color='green', alpha=0.4, edgecolor='black', linewidth=1.5, label='_nolegend_')

    ax.bar(0, 0, color='blue', alpha=0.6, edgecolor='black', linewidth=1.5, label='Real model')
    ax.bar(0, 0, color='red', alpha=0.6, edgecolor='black', linewidth=1.5, label='Tested model')
    if np.all(backgrounds!=np.zeros_like(backgrounds)):
        ax.bar(0, 0, color='green', alpha=0.6, edgecolor='black', linewidth=1.5, label='Background')


    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc = 'upper left', fontsize=18)
    ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=18)
    ax.set_ylabel('Counts', fontsize=18)

    chi2 = np.sum((N_real*branchings_real - N_tested*branchings_tested)**2/(N_tested*branchings_tested + backgrounds), axis = 0)

    ax.set_title(f'Comparison of two models $(N_{{real}} = {N_real:.1f}, N_{{tested}} = {N_tested:.1f})' + '$\n$\chi^2_{th} '+f'= {chi2:.1f}$', fontsize=18)
 


