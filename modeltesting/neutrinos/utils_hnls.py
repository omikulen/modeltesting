
"""
    Handles analysis of HNL model, defined as (N, xe, xmu) where N is the number of decays and xe, xmu are the branching ratios.
    Each decay channel is parametrized by the background b_i, and the effective branching ratio br_i(xe, xmu) 
    which includes the decay probability and the detection efficiency.

    We compare a real model (xe, xmu) outside the neutrino oscillation region with the tested models inside the allowed region.
    Given a prior probability distribution for the tested model, we require the combined CL = 0.9*prior for exclusion,
    and then search the number of events N that gives the required exclusion probability.

"""

import numpy as np
from pkg_resources import resource_filename
from ..utils_modeltest import find_optimal_Nreal, find_estimated_Nreal
from .utils_nufit import _h


def neutrino_priors(hierarchy):
    """


The function handles the extraction of neutrino priors. It returns a vectorized function that takes sets of \( (x_e, x_mu) \) and returns the corresponding priors. The resolutions of the priors are 400x400.

### Parameters

- `hierarchy` (string): The type of hierarchy (e.g., "NH" for Normal Hierarchy or "IH" for Inverted Hierarchy).

### Returns

- Vectorized function `_`: This function accepts (x_e, x_mu) in various formats.

## Nested Function `_(*args, grid=False)`

### Overview

The nested function `_` takes either one or two arguments to return the priors. It is designed to work with different input shapes and forms.

### Parameters

- `*args`: One or two arrays representing \( x_e \) and \( x_{\mu} \). Can be in the form \( (x_e, x_{\mu}) \) or \( ((x_e, x_{\mu})) \) of shape \( (N, 2) \) or \( (2, N) \).
- `grid` (boolean, optional): Whether to create a meshgrid from the inputs. Defaults to `False`.

### Returns

- Array of priors corresponding to the input \( x_e \) and \( x_{\mu} \).

### Example Usage

```python
>>> priors = neutrino_priors("NH")
>>> priors([0.05, 0.07, 0.1], 
           [0.3, 0.5, 0.7])
array([0.6492186 , 0.91551395, 0.40957517])
```


```python
>>> priors([[0.05, 0.07, 0.1], 
           [0.3, 0.5, 0.7]])
array([0.6492186 , 0.91551395, 0.40957517])
```


```python
>>> priors([[0.05, 0.3], 
         [0.07, 0.5], 
         [0.1 , 0.7]])
array([0.6492186 , 0.91551395, 0.40957517])
```

```python
>>> priors([[0.05, 0.07, 0.1], 
           [0.3, 0.5, 0.7]],
            grid = True)
array([[0.6492186 , 0.99956093, 0.9675604 ],
       [0.69489185, 0.91551395, 0.65906482],
       [0.91170954, 0.6787437 , 0.40957517]])
```

---


    """
    from scipy.interpolate import RegularGridInterpolator
    h = _h(hierarchy)
    path = resource_filename(__name__, f'data/nupriors_{h}.npy')

    nupriors_data = np.load(path)
    nupriors_data /= np.max(nupriors_data)
    func = RegularGridInterpolator(
        (np.linspace(0, 1, nupriors_data.shape[0]), 
         np.linspace(0, 1, nupriors_data.shape[1])), 
         nupriors_data, bounds_error=False, fill_value=0.0)
    # def nupriors(xe, xmu):
    #     return nupriors_data[int(xe*nupriors_data.shape[0]+1e-5), int(xmu*nupriors_data.shape[1]+1e-5)]
    # func = np.vectorize(nupriors)


    def _(*args, grid = False):
        if len(args) == 1:
            xy = np.array(args[0])
            if xy.shape[1] == 2:
                x, y = xy.T
            elif xy.shape[0] == 2:
                x, y = xy
        else:
            x, y = args
        if grid:
            x, y = np.meshgrid(x, y)
        return func(np.array((x, y)).T)
    
        
    return _





def br(x:np.ndarray, data, observe_photons = True):
    """
Handles the extraction of branching ratios from the txt file
    """
    if type(data) == str:
        with open(data, 'r') as f:
            for dw in f:
                if ',' in dw:
                    branching_data = dw.split(',')
                else:
                    branching_data = dw.split(' ')
                data = np.array(branching_data, dtype=float)
                break
    elif type(data) in [list, np.ndarray]:
        pass
    else:
        raise ValueError('data must be either a string or a list/np.ndarray')


    gamma_ehCC = data[1]
    gamma_muhCC = data[2]
    gamma_hNC = data[3]
    gamma_hNCnoph = data[4]
    gamma_emu = data[5]
    
    gamma_see = data[6]
    gamma_dee = data[7]
    gamma_smumu = data[8]
    gamma_dmumu = data[9]
    gamma_inv = data[10]
    xe, xmu = x
    xtau = 1 - xe - xmu
    Gamma_total = gamma_hNC + gamma_inv + gamma_ehCC*xe + gamma_muhCC*xmu 
    Gamma_total += gamma_see*xe + gamma_dee*(xmu+xtau) + gamma_smumu*xmu + gamma_dmumu*(xe+xtau)
    Gamma_total += gamma_emu*(xe+xmu)

    if observe_photons:
        return np.array([
            gamma_see*xe + gamma_dee*(xmu+xtau), #ee
            gamma_emu*(xe+xmu),  #emu
            gamma_smumu*xmu + gamma_dmumu*(xe+xtau), #mumu
            gamma_hNC, #hNC
            gamma_ehCC*xe, #ehCC
            gamma_muhCC*xmu, #muhCC
        ])/Gamma_total
    
    else:
        return np.array([
            gamma_see*xe + gamma_dee*(xmu+xtau), #ee
            gamma_emu*(xe+xmu),  #emu
            gamma_smumu*xmu + gamma_dmumu*(xe+xtau), #mumu
            # gamma_hNCnoph, #hNC without photons
            gamma_ehCC*xe, #ehCC
            gamma_muhCC*xmu, #muhCC
        ])/Gamma_total



def fake_model_scan(branching_function, 
               reference_point = np.array((0.5, 0.1)), 
               prior = lambda xe, xmu: np.exp(-(xe-0.2)**2/(0.1)**2),
               Nsamples = 100000,
               estimate_only = True,
               exclusion_probability = 0.9,
               exclusion_limit = 0.9,
               backgrounds = None,
               logs = False,
               **kwargs
               ):
    """
Scans over the fake models (xe, xmu) that are consistent with neutrino oscillations to find the required number of events to exclude the real model.
First step is to estimate the models which are the hardest to distingush from the real model.
Then we rescan over these model to find the exact value of the number of events.
    """
    number_of_channels = len(branching_function(reference_point))
    if backgrounds is None:
        backgrounds = np.zeros(number_of_channels)


    for key in ('xemin', 'xmumin'):
        if key not in kwargs.keys():
            kwargs[key] = 0.0

    for key in ('xemax',  'xmumax'):
        if key not in kwargs.keys():
            kwargs[key] = 1.0

    for key in ('xepoints', 'xmupoints'):
        if key not in kwargs.keys():
            kwargs[key] = 200

    # xe, xmu grids to scan over nu osc. compatible parameters.
        
    xe_grid = np.linspace(kwargs['xemin'], kwargs['xemax'], kwargs['xepoints'], endpoint=False)
    xmu_grid = np.linspace(kwargs['xmumin'], kwargs['xmumax'], kwargs['xmupoints'], endpoint=False)
    
    Nreal_grid = np.zeros((len(xmu_grid), len(xe_grid)))

    xmu_meshgrid, xe_meshgrid = np.meshgrid(xmu_grid, xe_grid)


    prior_grid = prior(xe_meshgrid, xmu_meshgrid)

    critical_point_data = {
        "Nreal" : 0,
        "tested_point" : (0, 0),
        "Ntested" : 0,
        "estimate" : True,
        "experimental_CL" : 0,
        "neutrino_prior" : 0,
    }
    
    # the `estimate_only` uses find_estimated_NrNf to find the Nr that gives the desired exclusion probability
    for i, xe in enumerate(xe_grid):
        for j, xmu in enumerate(xmu_grid):
            pr = prior_grid[i,j]

            # the neutrino oscillation region is where the prior is above the exclusion limit
            if pr > 1 - exclusion_limit:
                if np.linalg.norm(np.array((xe, xmu))-reference_point) > 0.01 and xe+xmu <= 1.0:
                    Nreal, Ntested = find_estimated_Nreal(branching_function(reference_point),
                                                    branching_function((xe, xmu)), 
                                                    CL= 1 - (1 - exclusion_limit)/pr)
                    
                    Nreal_grid[j, i] = Nreal
                    if Nreal > critical_point_data["Nreal"]:
                        critical_point_data["Nreal"] = Nreal
                        critical_point_data["tested_point"] = (xe, xmu)
                        critical_point_data["Ntested"] = Ntested
                        critical_point_data["experimental_CL"] = max(0, 1 - (1 - exclusion_limit)/pr)
                        critical_point_data["neutrino_prior"] = pr


    if logs:
        print("Finished estimates")
        print(f'Critical point data: {critical_point_data}\n')
    

    if not estimate_only:
        critical_point_data["estimate"] = False
        estimated_Nreal = critical_point_data["Nreal"]
        minval = np.min(Nreal_grid[Nreal_grid > 0])
        # if not estimate_only, we rescan points with largest estimated Nr (>0.7 of max Nr) to find the exact value
        critical_point_data = {
            "Nreal" : 0,
            "tested_point" : (0, 0),
            "Ntested" : 0,
            "experimental_CL" : 0,
            "neutrino_prior" : 0,
        }


        from tqdm import tqdm
        for i, xe in tqdm(enumerate(xe_grid)):
            for j, xmu in enumerate(xmu_grid):
                if Nreal_grid[j, i] > 0.7*(estimated_Nreal - minval) + minval:
                    pr = prior_grid[i,j]

                    if pr > 1 - exclusion_limit:
                        if np.linalg.norm(np.array((xe, xmu))-reference_point) > 0.01 and xe+xmu <= 1.0:                       
                            Nreal, Ntested, prob = find_optimal_Nreal(branchings_real = branching_function(reference_point),
                                                branchings_tested = branching_function((xe, xmu)),
                                                exclusion_probability = exclusion_probability,
                                                CL = max(0, 1 - (1 - exclusion_limit)/pr),
                                                Nsamples = Nsamples, convergence_rate=0.5,
                                                backgrounds = backgrounds,)
                            
                            if Nreal > critical_point_data["Nreal"]:
                                critical_point_data["Nreal"] = Nreal
                                critical_point_data["tested_point"] = (xe, xmu)
                                critical_point_data["Ntested"] = Ntested
                                critical_point_data["experimental_CL"] = max(0, 1 - (1 - exclusion_limit)/pr)
                                critical_point_data["neutrino_prior"] = pr
                                critical_point_data["chi2"] = np.sum((Ntested*branching_function((xe, xmu)) - Nreal*branching_function(reference_point))**2/(Ntested*branching_function((xe, xmu)) + backgrounds + 1e-10))
                                critical_point_data["exclusion_probability"] = prob

    if logs:
        print(f'The required number of events to exclude {reference_point} = {critical_point_data["Nreal"]}')
        print(f'The best-fit model is {critical_point_data["tested_point"]} with Nr = {critical_point_data["Nreal"]} and Nf = {critical_point_data["Ntested"]}')
        print(f'The required experimental CL = {critical_point_data["experimental_CL"]}, prior = {critical_point_data["neutrino_prior"]}\n')
    
    return critical_point_data, reference_point



    






def real_model_scan(branching_function, 
               reference_points = np.array(((0.5, 0.1),)), 
               prior = lambda xe, xmu: np.exp(-(xe-0.2)**2/(0.1)**2),
               Nsamples = 100000,
               estimate_only = True,
               exclusion_probability = 0.9,
               exclusion_limit = 0.9,
               backgrounds = None,
               logs = False):
    

    results = []
    for reference_point in reference_points:
        results.append(fake_model_scan(branching_function,reference_point=reference_point,prior=prior,Nsamples=Nsamples,estimate_only=estimate_only,exclusion_probability=exclusion_probability,exclusion_limit=exclusion_limit,backgrounds=backgrounds,logs=logs))
    return results
