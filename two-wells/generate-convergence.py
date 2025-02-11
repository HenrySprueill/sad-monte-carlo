#!/usr/bin/python3

import os, time
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import system, compute
import heat_capacity, styles
import glob
import time
from tqdm import tqdm

T = np.linspace(0.001,0.01,175)

E = np.linspace(-system.h_small, 0, 100000)
E = 0.5*(E[1:] + E[:-1])
dE = E[1] - E[0]
hist = None

lowest_interesting_E = -1.07
highest_interesting_E = -0.5

lowest_interestng_T=0.008

indices_for_err = np.array([i for i in range(len(E)) if lowest_interesting_E <= E[i] <= highest_interesting_E])
E_for_err = E[indices_for_err]

def normalize_S(S):
    S = S - max(S)
    total = np.sum(np.exp(S)*dE)
    return S - np.log(total)
if os.path.exists(system.name()+'-new.npz'):

    correct_S = normalize_S(system.S(E))

    correct_S_for_err = normalize_S(system.S(E[indices_for_err]))

    T,correct_C = heat_capacity.data(system.S)

    np.savez(system.name()+'-new.npz', E=E,
                            T=T,
                            correct_S=normalize_S(system.S(E)),
                            correct_S_for_err=correct_S_for_err,
                            correct_C=correct_C)
else:
    with np.load(system.name()+'.npz') as dat:
        correct_S_for_err = dat['correct_S_for_err']
        correct_C=dat['correct_C']
        correct_S = dat['correct_S']
        E = dat['E']

paths = []
for fname in sorted(glob.glob(os.path.join('thesis-data', '*+'+system.system+'*-lnw.dat'))):
    if not ('half-barrier' in fname):
        if 'sad' in fname:
            if True:#not( '0.01+0.001' in fname):
                paths.append(fname)
        elif ('wl' in fname or 'itwl' in fname) and '0.01' in fname:
            if True:#'0.001+' in fname:
                paths.append(fname)
        elif 'z+' in fname:
            paths.append(fname)
        else:
            paths.append(fname)

def generate_npz(fname):
    print(fname)
    base = fname[:-8]
    if os.path.exists(fname+'diag.npz'):
        pass
    method = base[:base.find('-')]

    energy_boundaries, mean_e, my_lnw, my_system, p_exc = compute.read_file(base)
    
    # Create a function for the entropy
    l_function, eee, sss = compute.linear_entropy(energy_boundaries, mean_e, my_lnw)
    S = normalize_S(l_function(E))


    T,C = heat_capacity.data(l_function, fname=fname)

    plt.figure('fraction-well')
    mean_which = np.loadtxt(f'{base}-which.dat')
    
    if os.path.exists(f'{base}-histogram.dat'):
        hist = np.loadtxt(f'{base}-histogram.dat')
    else:
        hist = []

    errors_S = []
    errors_C = []
    moves = []
    for frame_fname in tqdm(sorted(glob.glob(f'{base}/*-lnw.dat'))):
        frame_base =frame_fname[:-8]

        frame_moves = int(frame_fname[len(base)+1:-8])
        if frame_moves < 1e4:
            continue
        energy_boundaries, mean_e, my_lnw, my_system, p_exc = compute.read_file(frame_base)
        l_function, eee, sss = compute.linear_entropy(energy_boundaries, mean_e, my_lnw)
        moves.append(frame_moves)
        err = np.max(np.abs(normalize_S(l_function(E[indices_for_err])) - correct_S_for_err))
        errors_S.append(err)
        T_conv,C_conv = heat_capacity.data(l_function, fname=fname)
        C_mask = [t>=lowest_interestng_T for t in T_conv]
        errors_C.append(np.max(np.abs(correct_C[C_mask]-C_conv[C_mask])))

    
    np.savez(os.path.join(base)+'diag.npz',
                            E=E,
                            T=T,
                            mean_e=mean_e,
                            mean_which=mean_which,
                            hist=hist,
                            S=S,
                            C=C,
                            moves=moves,
                            errors_S=errors_S,
                            errors_C=errors_C)

def generate_npz_tempering(fname):
    print(fname)
    base = fname[:-1]
    if os.path.exists(fname+'.npz'):
        return
    method = base[:base.find('-')]
    

    errors_C = []
    moves = []
    for frame_fname in sorted(glob.glob(f'{base}/*.npz')):
        frame_base =frame_fname[:-4]


        frame_moves = int(frame_fname[len(base)+1:-4])
        frame_data = np.load(frame_fname)
        if frame_moves < 1e4:
            continue
        moves.append(frame_moves)
        t_low, t_peak, t_high= heat_capacity._set_temperatures()
        T = np.concatenate((t_low, t_peak, t_high))
        mean_E_func = interp1d(frame_data['T'], frame_data['mean_E'], fill_value='extrapolate', kind='cubic')
        mean_E_sqr_func = interp1d(frame_data['T'], frame_data['var_E'] + frame_data['mean_E']**2, fill_value='extrapolate', kind='cubic')
        C = (mean_E_sqr_func(T) - mean_E_func(T)**2)/(T**2)
        C_mask = [t>=lowest_interestng_T for t in T]
        assert(len(correct_C) == len(C))
        errors_C.append(np.max(np.abs(correct_C[C_mask]-C[C_mask])))
    
    np.savez(os.path.join('.',base)+'+plt.npz',
                            T=T,
                            C=C,
                            moves=moves,
                            errors_C=errors_C)

from multiprocessing import Pool

if __name__ == '__main__':
    pass
    with Pool(8) as p:
        p.map(generate_npz, list(paths))