import numpy as np
deg = np.pi/180

## dunno wat this is
# def _hnl_data(**kwargs):
#     return hnl_data(kwargs, kwargs, hierarchy = kwargs['hierarchy'])

bestfitNH = {
    'theta12': 33.41*deg,
    'theta23': 49.1*deg,
    'theta13': 8.54*deg,
    'delta': 197*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.511e-3,
}
bestfitIH = {
    'theta12': 33.41*deg,
    'theta13': 8.57*deg,
    'theta23': 49.5*deg,
    'delta': 286*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.498e-3,
}

def _h(hierarchy):
    if hierarchy.lower() in ['normal', 'nh', 'n', 'no']:
        return 'n'
    elif hierarchy.lower() in ['inverse', 'ih', 'i', 'in']:
        return 'i'
    else:
        raise ValueError('Hierarchy must be either normal (NO, NH, N) or inverse (IO, IH, I)')



def PMNS(**kwargs):

    theta12 = kwargs['theta12']
    theta13 = kwargs['theta13']
    theta23 = kwargs['theta23']
    delta = kwargs['delta']
    eta1 = kwargs.get('eta1', 0)
    eta2 = kwargs.get('eta2', 0)
    eta3 = kwargs.get('eta3', 0)

    R1 = np.array([
        [1, 0, 0],
        [0, np.cos(theta23), np.sin(theta23)],
        [0, -np.sin(theta23), np.cos(theta23)]
    ])

    R2 = np.array([
        [np.cos(theta13), 0, np.sin(theta13)*np.exp(-1j*delta)],
        [0, 1, 0],
        [-np.sin(theta13)*np.exp(1j*delta), 0, np.cos(theta13)]
    ])

    R3 = np.array([
        [np.cos(theta12), np.sin(theta12), 0],
        [-np.sin(theta12), np.cos(theta12), 0],
        [0, 0, 1]
    ])

    Reta = np.diag([np.exp(1j*eta1), np.exp(1j*eta2), np.exp(1j*eta3)])

    return R1@R2@R3@Reta





def CasasIbarra_2(*args, hierarchy = 'n', **kwargs):
    hierarchy = _h(hierarchy)

    nu_pars = bestfitIH if hierarchy == 'i' else bestfitNH


    if args:
        if isinstance(args[0], dict):
            nu_pars.update(args[0])
        else:
            raise ValueError("Invalid argument type. Only a dictionary is allowed as a positional argument.")
    nu_pars.update(kwargs)


    # Neutrino parameters (3 theta, delta_CP, 2 dm2) are contained in nu_pars
    theta12 = nu_pars['theta12']
    theta13 = nu_pars['theta13']
    theta23 = nu_pars['theta23']
    delta = nu_pars['delta']
    dm2sol = nu_pars['dm2sol']
    dm2atm = nu_pars['dm2atm']

    # Extra parameters that are not constrained by neutrino data are contained in extra_pars
    # these include neutrino majorana phases and the omega angle
    eta = nu_pars['eta']
    reomega = nu_pars['reomega']
    imomega = nu_pars['imomega']


    V_PMNS = PMNS(theta12 = theta12, theta13 = theta13, theta23 = theta23, delta = delta, eta2 = eta)

    if hierarchy == 'n':
        m1 = 0
        m2 = np.sqrt(dm2sol)
        m3 = np.sqrt(dm2atm + dm2sol)
        
        RHNL = np.array([
            [0, 0],
            [np.cos(reomega + 1j*imomega), np.sin(reomega + 1j*imomega)],
            [-np.sin(reomega + 1j*imomega), np.cos(reomega + 1j*imomega)],
        ], dtype = complex)


    elif hierarchy == 'i':
        m1 = np.sqrt(dm2atm)
        m2 = np.sqrt(dm2atm + dm2sol)
        m3 = 0
        
        RHNL = np.array([
            [np.cos(reomega + 1j*imomega), np.sin(reomega + 1j*imomega)],
            [-np.sin(reomega + 1j*imomega), np.cos(reomega + 1j*imomega)],
            [0, 0]
        ], dtype = complex)

    mnu = np.diag([m1, m2, m3])
    Vm = 1j*(V_PMNS@(np.sqrt(mnu))@RHNL)
    
    ### quick computation in the approximate 
    # for i in range(3):
    #     if hierarchy == 'normal':
    #         u_alpha[i] = m2*np.abs(V_PMNS[i, 1])**2 + m3*np.abs(V_PMNS[i, 2])**2 - 2*np.sqrt(m2*m3)*np.imag(V_PMNS[i, 1]*np.conj(V_PMNS[i, 2])*np.exp(-1j*eta))            
    #     elif hierarchy == 'inverse':
    #         u_alpha[i] = m1*np.abs(V_PMNS[i, 0])**2 + m2*np.abs(V_PMNS[i, 1])**2 - 2*np.sqrt(m1*m2)*np.imag(V_PMNS[i, 0]*np.conj(V_PMNS[i, 1])*np.exp(-1j*eta))
    #     else:
    #         raise ValueError('Hierarchy must be either normal or inverse')
        
    return np.square(Vm)
    

import numpy as np
deg = np.pi/180

## dunno wat this is
# def _hnl_data(**kwargs):
#     return hnl_data(kwargs, kwargs, hierarchy = kwargs['hierarchy'])

bestfitNH = {
    'theta12': 33.41*deg,
    'theta23': 49.1*deg,
    'theta13': 8.54*deg,
    'delta': 197*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.511e-3,
}
bestfitIH = {
    'theta12': 33.41*deg,
    'theta13': 8.57*deg,
    'theta23': 49.5*deg,
    'delta': 286*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.498e-3,
}

def _h(hierarchy):
    if hierarchy.lower() in ['normal', 'nh', 'n', 'no']:
        return 'n'
    elif hierarchy.lower() in ['inverse', 'ih', 'i', 'in']:
        return 'i'
    else:
        raise ValueError('Hierarchy must be either normal (NO, NH, N) or inverse (IO, IH, I)')



def PMNS(**kwargs):

    theta12 = kwargs['theta12']
    theta13 = kwargs['theta13']
    theta23 = kwargs['theta23']
    delta = kwargs['delta']
    eta1 = kwargs.get('eta1', 0)
    eta2 = kwargs.get('eta2', 0)
    eta3 = kwargs.get('eta3', 0)

    R1 = np.array([
        [1, 0, 0],
        [0, np.cos(theta23), np.sin(theta23)],
        [0, -np.sin(theta23), np.cos(theta23)]
    ])

    R2 = np.array([
        [np.cos(theta13), 0, np.sin(theta13)*np.exp(-1j*delta)],
        [0, 1, 0],
        [-np.sin(theta13)*np.exp(1j*delta), 0, np.cos(theta13)]
    ])

    R3 = np.array([
        [np.cos(theta12), np.sin(theta12), 0],
        [-np.sin(theta12), np.cos(theta12), 0],
        [0, 0, 1]
    ])

    Reta = np.diag([np.exp(1j*eta1), np.exp(1j*eta2), np.exp(1j*eta3)])

    return R1@R2@R3@Reta





def CasasIbarra_2(*args, hierarchy = 'n', **kwargs):
    hierarchy = _h(hierarchy)

    nu_pars = bestfitIH if hierarchy == 'i' else bestfitNH


    if args:
        if isinstance(args[0], dict):
            nu_pars.update(args[0])
        else:
            raise ValueError("Invalid argument type. Only a dictionary is allowed as a positional argument.")
    nu_pars.update(kwargs)


    # Neutrino parameters (3 theta, delta_CP, 2 dm2) are contained in nu_pars
    theta12 = nu_pars['theta12']
    theta13 = nu_pars['theta13']
    theta23 = nu_pars['theta23']
    delta = nu_pars['delta']
    dm2sol = nu_pars['dm2sol']
    dm2atm = nu_pars['dm2atm']

    # Extra parameters that are not constrained by neutrino data are contained in extra_pars
    # these include neutrino majorana phases and the omega angle
    eta = nu_pars['eta']
    reomega = nu_pars['reomega']
    imomega = nu_pars['imomega']


    V_PMNS = PMNS(theta12 = theta12, theta13 = theta13, theta23 = theta23, delta = delta, eta2 = eta)

    if hierarchy == 'n':
        m1 = 0
        m2 = np.sqrt(dm2sol)
        m3 = np.sqrt(dm2atm + dm2sol)
        
        RHNL = np.array([
            [0, 0],
            [np.cos(reomega + 1j*imomega), np.sin(reomega + 1j*imomega)],
            [-np.sin(reomega + 1j*imomega), np.cos(reomega + 1j*imomega)],
        ], dtype = complex)

        
    elif hierarchy == 'i':
        m1 = np.sqrt(dm2atm)
        m2 = np.sqrt(dm2atm + dm2sol)
        m3 = 0
        
        RHNL = np.array([
            [np.cos(reomega + 1j*imomega), np.sin(reomega + 1j*imomega)],
            [-np.sin(reomega + 1j*imomega), np.cos(reomega + 1j*imomega)],
            [0, 0]
        ], dtype = complex)

    mnu = np.diag([m1, m2, m3])
    Vm = 1j*(V_PMNS@(np.sqrt(mnu))@RHNL)
    
    ### quick computation in the approximate 
    # for i in range(3):
    #     if hierarchy == 'normal':
    #         u_alpha[i] = m2*np.abs(V_PMNS[i, 1])**2 + m3*np.abs(V_PMNS[i, 2])**2 - 2*np.sqrt(m2*m3)*np.imag(V_PMNS[i, 1]*np.conj(V_PMNS[i, 2])*np.exp(-1j*eta))            
    #     elif hierarchy == 'inverse':
    #         u_alpha[i] = m1*np.abs(V_PMNS[i, 0])**2 + m2*np.abs(V_PMNS[i, 1])**2 - 2*np.sqrt(m1*m2)*np.imag(V_PMNS[i, 0]*np.conj(V_PMNS[i, 1])*np.exp(-1j*eta))
    #     else:
    #         raise ValueError('Hierarchy must be either normal or inverse')
        
    return np.square(Vm)
    
