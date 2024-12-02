import scipy.io 
import numpy as np
from plots_FHN import plot_phase,plot_fhn,plot_error_fhn
from plots_HH import plot_hh, plot_error_hh
import matplotlib.pyplot as plt
import sys
import os

 
model = sys.argv[1].lower()
type_of_plot = sys.argv[2].lower()
n_exmaples = int(sys.argv[3])

if model == 'fhn':

    path_load = os.getcwd() + '/../../data/' + model + '/fhn_trainL2_n_375_points_1260_tf_100.mat'

    print(f'Loading data from {path_load.split('/../',1)[1]}')
    data_load = scipy.io.loadmat(path_load)
    
    I_app   = data_load["input"]
    V_exact = data_load["V_exact"]
    w_exact = data_load["w_exact"]
    V_pred  = data_load["V_pred"]
    w_pred  = data_load["w_pred"]

    t = np.linspace(0,100,1260)

    indx_gen = np.random.randint(10,295,n_exmaples)#general examples
    indx_nap = np.random.randint(296,325,n_exmaples)#nap examples
    indx_tf  = np.random.randint(325,len(V_exact),n_exmaples)#t_fin examples 

    if type_of_plot == 'phase':
        plot_phase(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_gen)
        plot_phase(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_nap)
        plot_phase(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_tf)
    
    elif type_of_plot == 'multiple':
        plot_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_gen)
        plot_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_nap)
        plot_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_tf)
    
    elif type_of_plot == 'error':
        plot_error_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_gen)
        plot_error_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_nap)
        plot_error_fhn(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_tf)

    else:
        raise NameError(f'The type of plots can be either phase or error, and not {type_of_plot}')

if model == 'hh':

    path_load = os.getcwd() + '/../../data/' + model + '/hh_trainL2_n_375_points_1260_tf_100.mat'

    print(f'Loading data from {path_load.split('/../',1)[1]}')
    data_load = scipy.io.loadmat(path_load)
    
    I_app   = data_load["input"]
    V_exact = data_load["V_exact"]
    m_exact = data_load["m_exact"]
    h_exact = data_load["h_exact"]
    n_exact = data_load["n_exact"]
    
    V_pred  = data_load["V_pred"]
    m_pred  = data_load["m_pred"]
    h_pred  = data_load["h_pred"]
    n_pred  = data_load["n_pred"]

    t = np.linspace(0,100,1260)

    indx_gen = np.random.randint(10,285,n_exmaples)#general examples
    indx_nap = np.random.randint(286,316,n_exmaples)#nap examples
    indx_ihigh = np.random.randint(317,347,n_exmaples)#nap examples
    indx_tf  = np.random.randint(347,len(V_exact),n_exmaples)#t_fin examples 

    if type_of_plot == 'multiple':
   
        plot_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_gen)
        plot_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_nap)
        plot_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_ihigh)
        plot_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_tf)
   
    elif type_of_plot == 'error':
        plot_error_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_gen)
        plot_error_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_nap)
        plot_error_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_ihigh)
        plot_error_hh(V_exact,m_exact,h_exact,n_exact,V_pred,m_pred,h_pred,n_pred,I_app,t,indx_tf)
    else:
        raise NameError(f'The type of plots can be either multiple or error, and not {type_of_plot}')
# plot_error(V_exact,w_exact,V_pred,w_pred,I_app,t,indx_gen)