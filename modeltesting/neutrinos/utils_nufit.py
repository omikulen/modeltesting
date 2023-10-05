import numpy as np
import warnings

deg = np.pi/180


NuFITNH = {
    'theta12': 33.41*deg,
    'theta13': 8.54*deg,
    'theta23': 49.1*deg,
    'delta': 197*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.511e-3,
}

NuFITIH = {
    'theta12': 33.41*deg,
    'theta13': 8.57*deg,
    'theta23': 49.5*deg,
    'delta': 286*deg,
    'dm2sol': 7.41e-5,
    'dm2atm': 2.498e-3,
}

def _h(hierarchy):
    """
Uniforms the hierarchy input.
    """
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





def CasasIbarra_2(*args, hierarchy = None, **kwargs):
    """
    Calculates the matrix M_I theta^2_{alpha I} (in eV) for two additional sterile neutrinos I=1,2,
    according to the Casas-Ibarra parameterization. The lightest neutrino mass is assumed to be zero.

    Parameters:
    -----------
    hierarchy : str, optional
        Specifies the neutrino mass hierarchy. Accepts 'n' for normal hierarchy and 'i' for inverted hierarchy.
        Default is 'n'.
    
    *args : dict, optional
        A dictionary containing neutrino parameters. Only a dictionary is allowed as a positional argument.
    
    **kwargs : dict
        Additional neutrino parameters specified as keyword arguments. These will update or add to the parameters
        specified in args.

    Neutrino Parameters in args or kwargs. 
    --------------------------------------
    theta12, theta13, theta23 : float
        Mixing angles in rad.
    delta : float
        CP-violating phase in rad.
    dm2sol, dm2atm : float
        Solar and atmospheric squared mass differences in eV^2.

        
    Extra Parameters for Casas-Ibarra Parameterization. **Must** be specified in args or kwargs:
    --------------------------------------
    eta : float
        Neutrino Majorana phase in rad.
    reomega, imomega : float
        Real and imaginary parts of the complex omega angle.

    Returns:
    --------
    np.ndarray
        Matrix theta^2_{alpha I} M_I (in eV), with shape (3, 2).

    Example:
    --------
    >>> CasasIbarra_2({'theta23':0.842, 'delta':3.87, 'dm2atm':2.523e-3},
                        hierarchy = 'i', 
                        eta = np.pi/2, imomega = 2, reomega = np.pi/2)
    array([[ 1.28454939e+00+9.42935293e-17j, -1.30366463e+00-9.61437203e-17j],
        [ 6.52497722e-02+4.56574335e-02j, -5.11429167e-02-4.13131453e-02j],
        [ 6.38885718e-03-1.66738902e-02j, -2.63450905e-04+1.18983009e-02j]])

    """
    if hierarchy is None:
        warnings.warn('Hierarchy not specified. Assuming normal hierarchy (NO)')
        hierarchy = 'n'
    hierarchy = _h(hierarchy)

    nu_pars = NuFITIH.copy() if hierarchy == 'i' else NuFITNH.copy()


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
    try:
        eta = nu_pars['eta']
        reomega = nu_pars['reomega']
        imomega = nu_pars['imomega']
    except KeyError:
        raise KeyError('Missing required parameter. eta, reomega, and imomega must be specified.')

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
    
    ### quick computation in the approximate lepton symmetry limit
    # for i in range(3):
    #     if hierarchy == 'normal':
    #         u_alpha[i] = m2*np.abs(V_PMNS[i, 1])**2 + m3*np.abs(V_PMNS[i, 2])**2 - 2*np.sqrt(m2*m3)*np.imag(V_PMNS[i, 1]*np.conj(V_PMNS[i, 2])*np.exp(-1j*eta))            
    #     elif hierarchy == 'inverse':
    #         u_alpha[i] = m1*np.abs(V_PMNS[i, 0])**2 + m2*np.abs(V_PMNS[i, 1])**2 - 2*np.sqrt(m1*m2)*np.imag(V_PMNS[i, 0]*np.conj(V_PMNS[i, 1])*np.exp(-1j*eta))
    #     else:
    #         raise ValueError('Hierarchy must be either normal or inverse')
        
    return np.square(Vm)
    

def CasasIbarra_3(*args, hierarchy = None, **kwargs):
    """
    Similar to :func:`CasasIbarra_2`, but for 3 sterile neutrinos.

    Extra Parameters for Casas-Ibarra Parameterization. **Must** be specified in args or kwargs:
    -----------
    mnu0 : float
        The lightest neutrino mass in eV.
    eta1, eta2 : float
        Neutrino Majorana phases in rad.
    reomega1, imomega1, reomega2, imomega2, reomega3, imomega3 : float
        Real and imaginary parts of the three complex omega angles.
    
    For other parameters, see :func:`CasasIbarra_2`.

    Returns:
    --------
    np.ndarray
        Matrix theta^2_{alpha I} M_I (in eV), with shape (3, 3).

    Example:
    --------
    >>> CasasIbarra_3({'theta23':0.842, 'delta':3.87, 'dm2atm':2.523e-3, 'eta1':0.5, 'eta2':1.5},
                        hierarchy = 'i', 
                        mnu0 = 0.1,
                        imomega1 = 4, reomega1 = 0.1, 
                        imomega2 = -2, reomega2 = 0.3, 
                        imomega3 = 4, reomega3 = 0.9)
    array([[ 1.15141141e+05-3.96734150e+05j, -1.15167947e+05+3.96264529e+05j,
            2.67973657e+01+4.69554113e+02j],
        [ 5.69731854e+03-4.89716385e+03j, -5.69344980e+03+4.88997677e+03j,
            -3.89092557e+00+7.18107712e+00j],
        [ 7.69547591e+05-1.31355168e+06j, -7.69162199e+05+1.31184675e+06j,
            -3.85406852e+02+1.70490113e+03j]])
    """

    if hierarchy is None:
        warnings.warn('Hierarchy not specified. Assuming normal hierarchy (NO)')
        hierarchy = 'n'
    hierarchy = _h(hierarchy)

    nu_pars = NuFITIH.copy() if hierarchy == 'i' else NuFITNH.copy()


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
    mnu0 = nu_pars['mnu0']
    eta1 = nu_pars['eta1']
    eta2 = nu_pars['eta2']

    reomega1 = nu_pars['reomega1']
    imomega1 = nu_pars['imomega1']
    reomega2 = nu_pars['reomega2']
    imomega2 = nu_pars['imomega2']
    reomega3 = nu_pars['reomega3']
    imomega3 = nu_pars['imomega3']

    V_PMNS = PMNS(theta12 = theta12, theta13 = theta13, theta23 = theta23, delta = delta, eta1 = eta1, eta2 = eta2)
    RHNL = PMNS(theta12 = reomega1 + 1j*imomega1, 
                theta13 = reomega2 + 1j*imomega2, 
                theta23 = reomega3 + 1j*imomega3, 
                delta = 0)
    
    if hierarchy == 'n':
        m1 = mnu0
        m2 = np.sqrt(dm2sol + mnu0**2)
        m3 = np.sqrt(dm2atm + dm2sol + mnu0**2)
    elif hierarchy == 'i':
        m1 = np.sqrt(dm2atm + mnu0**2)
        m2 = np.sqrt(dm2atm + dm2sol + mnu0**2)
        m3 = mnu0

    mnu = np.diag([m1, m2, m3])
    Vm = 1j*(V_PMNS@(np.sqrt(mnu))@RHNL)

    return np.square(Vm)