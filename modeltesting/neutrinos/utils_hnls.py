
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



def HNL_decaywidths(M:float):
    """
Precomputed decay widths for HNLs with mass M in GeV.
The data covers the range 0.1 GeV < M < 10 GeV.

### Parameters
    M (float): The mass of the HNL in GeV.

### Returns
    dict(str: (float, str))
        Dictionary of pairs \{decay channel:(decay width in GeV , mixings)\} for the HNL with mass M. The mixings are given as a string of 'e', 'm', 't' for electron, muon, tau.
    """

    path = resource_filename(__name__, f'data/decay_widths.txt')

    # handle out-masses
    if M < 0:
        raise ValueError('Mass must be positive')

    # go through the file to find the first line above the mass
    # since I know that the grid is np.logspace(-1, 1, 200), I hard-code the index
    # cause why not, I'm the boss here :D
    rough_index = min(np.log10(M/0.1)*100-5, 195)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < rough_index:
                continue
            if line.startswith('#'):
                continue
            else:
                if float(line[:5]) > M:
                    break

    # read the decay widths and renormalize them
    decay_data = line.split(',')
    decay_data = np.array(decay_data, dtype=float)*float(decay_data[0])**5/(392*np.pi**3*(246.22)**4)

    output = {
        '3nu': (decay_data[1], 'emt'),
        'nuee(e)': (decay_data[2], 'e'),
        'nuee(NC)': (decay_data[3], 'mt'),
        'nuemu': (decay_data[4], 'em'),
        'numumu(mu)': (decay_data[5], 'm'),
        'numumu(NC)': (decay_data[6], 'et'),
        'nuetau': (decay_data[7], 'et'),
        'numutau': (decay_data[8], 'mt'),
        'nutautau(tau)': (decay_data[9], 't'),
        'nutautau(NC)': (decay_data[10], 'em'),
        'nupi': (decay_data[11], 'emt'),
        'epi': (decay_data[12], 'e'),
        'mupi': (decay_data[13], 'm'),
        'nupipi': (decay_data[14], 'emt'),
        'epipi': (decay_data[15], 'e'),
        'mupipi': (decay_data[16], 'm'),
        'eK': (decay_data[17], 'e'),
        'muK': (decay_data[18], 'm'),
        'nueta': (decay_data[19], 'emt'),
        'nuetaprime': (decay_data[20], 'emt'),
        'nuomega': (decay_data[21], 'emt'),
        'nuphi': (decay_data[22], 'emt'),
        'nuqq': (decay_data[23], 'emt'),
        'eqq': (decay_data[24], 'e'),
        'muqq': (decay_data[25], 'm'),
        'tauqq': (decay_data[26], 't')
    }
    return output


def HNL_branchings(M:float):
    """
Branching ratios for HNLs with mass M in GeV.

### Parameters
    M (float): The mass of the HNL in GeV.

### Returns
    function(xe, xmu) -> dict
        function that takes the mixing angles and returns a dictionary of branching ratios. Accepts input formats func(xe, xmu); func([xe, xmu]); func(xe, xmu); func([xe, xmu, xtau]). 

### Notes
    The data covers mass range: 0.1 GeV < M < 10 GeV. Note that there are decays into both quarks and mesons, so the channels overlap. 
    At masses M>2, semileptonic decay widths are computed by decays into quarks; at masses M<2 - by decays into mesons. Automatically renormalize xe, xmu, xtau to xe+xmu+xtau = 1, if all three parameters are given. 
    
### Example Usage
```python
>>> br_1GeV = HNL_branchings(1.0)
>>> br_1GeV(0.5, 0.1)
    {'3nu': 0.16977063986554167,
    'nuemu': 0.09381525558969832,
    'nuetau': 0.0,
    'numutau': 0.0,
    'nupi': 0.1572076125154916,
    'epi': 0.1487190805222145,
    'mupi': 0.028759146393222756,
    'nupipi': 0.05330798091778008,
    'epipi': 0.17656146546016332,
    'mupipi': 0.03184897203877562,
    'eK': 0.0068757109145544375,
    'muK': 0.0013072339269646707,
    'nueta': 0.032595962854183994,
    'nuetaprime': 0.0013581651189243332,
    'nuomega': 0.013072339269646707,
    'nuphi': 0.0,
    'nuqq': 0.33325976605605834,
    'eqq': 0.33954127973108333,
    'muqq': 0.06252652666247899,
    'tauqq': 0.0,
    'nuee': 0.06060811843199837,
    'numumu': 0.02419231618083969,
    'nutautau': 0.0}
    """
    decay_data = HNL_decaywidths(M)

    def _(*args):
        if len(args) == 1:
            x = args[0]
        else:
            x = np.array(args)
        if len(x) == 2:
            xe, xmu = x
            xtau = 1 - xe - xmu
        elif len(x) == 3:
            xe, xmu, xtau = (x/np.sum(x))
        
        if xe < 0 or xmu < 0 or xtau < 0:
            raise ValueError('Mixing angle ratios x_alpha = U_alpha^2/U^2 must be non-negative')
        
        scaling = lambda s: (xe if 'e' in s else 0)+(xmu if 'm' in s else 0)+(xtau if 't' in s else 0)

        output = {key : val[0]*scaling(val[1]) 
                  for key, val in decay_data.items() }
        
        for l in ('e', 'mu', 'tau'):
            output[f'nu{l}{l}'] = output[f'nu{l}{l}({l})']+output[f'nu{l}{l}(NC)']
            output.pop(f'nu{l}{l}({l})')
            output.pop(f'nu{l}{l}(NC)')

        # total_width = np.sum(list(output.values()))
        leptonic_width = output['3nu']+output['nuee']+output['nuemu']+output['numumu']+output['nuetau']+output['numutau']+output['nutautau']
        if M > 2:
            hadronic_width = output['nuqq']+output['eqq']+output['muqq']+output['tauqq']
        else:
            hadronic_width = output['nupi']+output['nupipi']+output['nueta']+output['nuetaprime']+output['nuomega']+output['nuphi']+output['epi']+output['epipi']+output['eK']+output['mupi']+output['mupipi']+output['muK']

        total_width = leptonic_width + hadronic_width

        output = {key : val/total_width for key, val in output.items()}

        return output

    return _


def HNL_branchings_specific(M:float):
    """
Branching ratios for HNLs with mass M in GeV. Same to `HNL_branchings`, but with the branching ratios grouped into the following categories:
'ee', 'emu', 'mumu', 'NC', 'eCC', 'muCC'.

### Parameters
    M (float): The mass of the HNL in GeV.

### Returns
    function(xe, xmu) -> dict
        function that takes the mixing angles and returns a dictionary of branching ratios. See `HNL_branchings` for more details.
    """
    func = HNL_branchings(M)
    def _(*args):
        output = func(*args)
        output_specific = {
            'ee': output['nuee'],
            'emu': output['nuemu'],
            'mumu': output['numumu'],
            'NC': output['nuqq'] if M > 2 else (output['nupi']+output['nupipi']+output['nueta']+output['nuetaprime']+output['nuomega']+output['nuphi']),
            'eCC': output['eqq'] if M > 2 else (output['epi']+output['epipi']+output['eK']),
            'muCC': output['muqq'] if M > 2 else (output['mupi']+output['mupipi']+output['muK']),
        }
        return output_specific
    
    return _
        
    




def fake_model_scan(branching_function, 
               reference_point = np.array((0.5, 0.1)), 
               prior = lambda xe, xmu: np.exp(-(xe-0.2)**2/(0.1)**2 - (xmu-0.5)**2/(0.5)**2),
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



    if 'scanfile' in kwargs.keys():
        scanpoints = np.loadtxt(kwargs['scanfile'])
        xe_grid = scanpoints[:,0]
        xmu_grid = scanpoints[:,1]

    else:
        # xe, xmu grids to scan over nu osc. compatible parameters.
            
        xe_grid = np.linspace(kwargs['xemin'], kwargs['xemax'], kwargs['xepoints'], endpoint=False)
        xmu_grid = np.linspace(kwargs['xmumin'], kwargs['xmumax'], kwargs['xmupoints'], endpoint=False)
        

        xmu_meshgrid, xe_meshgrid = np.meshgrid(xmu_grid, xe_grid)
        xe_grid = xe_meshgrid.flatten()
        xmu_grid = xmu_meshgrid.flatten()
        
    Nreal_grid = np.zeros(len(xe_grid))
    prior_grid = prior(xe_grid, xmu_grid)

    critical_point_data = {
        "Nreal" : 0,
        "tested_point" : (0, 0),
        "Ntested" : 0,
        "estimate" : True,
        "experimental_CL" : 0,
        "neutrino_prior" : 0,
    }
    
    # the `estimate_only` uses find_estimated_NrNf to find the Nr that gives the desired exclusion probability
    for i, (xe, xmu, pr) in enumerate(zip(xe_grid, xmu_grid, prior_grid)):


        # the neutrino oscillation region is where the prior is above the exclusion limit
        if pr > 1 - exclusion_limit:
            if np.linalg.norm(np.array((xe, xmu))-reference_point) > 0.01 and xe+xmu <= 1.0:
                br_real = branching_function(reference_point)
                br_tested = branching_function((xe, xmu))
                if type(br_real) == dict:
                    br_real = np.array(list(br_real.values()))
                    br_tested = np.array(list(br_tested.values()))
                Nreal, Ntested = find_estimated_Nreal(branchings_real=br_real,
                                                branchings_tested=br_tested, 
                                                CL= 1 - (1 - exclusion_limit)/pr)
                
                Nreal_grid[i] = Nreal
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


        for i, (xe, xmu, pr) in enumerate(zip(xe_grid, xmu_grid, prior_grid)):
            if Nreal_grid[i] > 0.7*(estimated_Nreal - minval) + minval:

                if pr > 1 - exclusion_limit:
                    if np.linalg.norm(np.array((xe, xmu))-reference_point) > 0.01 and xe+xmu <= 1.0:
                        br_real = branching_function(reference_point)
                        br_tested = branching_function((xe, xmu))
                        if type(br_real) == dict:
                            br_real = np.array(list(br_real.values()))
                            br_tested = np.array(list(br_tested.values()))               
                        Nreal, Ntested, prob = find_optimal_Nreal(branchings_real = br_real,
                                            branchings_tested = br_tested,
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
                            critical_point_data["chi2"] = np.sum((Ntested*br_tested - Nreal*br_real)**2/(Ntested*br_tested + backgrounds + 1e-10))
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
