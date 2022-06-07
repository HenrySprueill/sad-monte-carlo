import os
import numpy as np
import glob
import system
import styles
import heat_capacity
from results_plotting import Results

import warnings
warnings.filterwarnings("ignore")


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams.update({'font.size': 16})


def legend_handles(exact = False):
    lines=[]
    if exact:
        exact_line = mlines.Line2D([], [], color='blue',
                            label='exact', linestyle=':')
        lines+=[exact_line]
    sad_line = mlines.Line2D([], [], color='k', marker='s',
                          markersize=8, label=r'SAD')
    itwl_line = mlines.Line2D([], [], color='k', marker='<',
                          markersize=8, label=r'$1/t$-WL')
    #z_line = mlines.Line2D([], [], color='k', marker='o',
    #                      markersize=8, label=r'ZMC')
    zero_line = mlines.Line2D([], [], color='g', linestyle='-',
                          markersize=12, label=r'$E_{barr}$=0.0')
    one_ten_line = mlines.Line2D([], [], color='tab:orange', linestyle='dashed',
                          markersize=12, label=r'$E_{barr}$=0.1')
    two_ten_line = mlines.Line2D([], [], color='tab:cyan', linestyle='-.',
                          markersize=12, label=r'$E_{barr}$=0.2')
    lines+=[sad_line, itwl_line,  zero_line, one_ten_line, two_ten_line]
    return lines

lowest_interesting_E = -1.1
highest_interesting_E = -0.5

lowest_interesting_T=0.008

exact = np.load(os.path.join('.',system.name()+'-new.npz'))
correct_S=exact['correct_S']
E=exact['E']
T=exact['T']
correct_C=exact['correct_C']
dE = E[1] - E[0]
paths = glob.glob(os.path.join('thesis-data-new','*.npz'))

subplot_fig, axs = plt.subplot_mosaic([
                            ['(a)','(b)'],
                            ['(c)','(d)']
                        ], 
                        tight_layout = True, 
                        figsize=(13,9),
                        )

plt.figure('latest-entropy')
plt.plot(E, correct_S, ':', label='exact', linewidth=2)
axs['(c)'].plot(E, correct_S, ':', label='exact', linewidth=2)

fig, ax = plt.subplots(figsize=[5, 4], num='latest-heat-capacity')
axins = ax.inset_axes( 0.5 * np.array([1, 1, 0.47/0.5, 0.47/0.5]))#[0.005, 0.012, 25, 140])
axins_subplot = axs['(d)'].inset_axes( 0.5 * np.array([1, 1, 0.47/0.5, 0.47/0.5]))

heat_capacity.plot_from_data(exact['T'],exact['correct_C'], ax=ax, axins=axins)
heat_capacity.plot_from_data(exact['T'],exact['correct_C'], ax=axs['(d)'], axins=axins_subplot)
ax.indicate_inset_zoom(axins, edgecolor="black")
axs['(d)'].indicate_inset_zoom(axins_subplot, edgecolor="black")

#Combines two strings, truncating the longer one
#to the length of the shorter one and adding them


z_filter = lambda fname: 'z+' in fname
tem_filter = lambda fname: 'tem+' in fname
seed_filter = lambda fname: 'seed-1+' in fname
step_filter = lambda fname, step: any(['de-1e-05+step-'+str(s) in fname for s in step])
suffix_filter = lambda fname: 'diag' in fname or 'plt' in fname
barrier1_filter = lambda fname: 'barrier-1+' not in fname and 'barrier-1-small' not in fname #and 'barrier-1e-1' not in fname
step = [0.01, 0.001, 0.0001]
total_filter = lambda fname: (step_filter(fname, step)) and seed_filter(fname) and suffix_filter(fname) and barrier1_filter(fname)
results = Results(E=E, S=correct_S, T=T, C=correct_C)
for fname in filter(total_filter, paths):
    print(fname)
    tail = fname[:fname.find('seed')]
    b = fname[fname.find('seed'):]
    i= fname.find('seed') + b.find('+')
    front = fname[i:]

    
    for seed in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678]:
        if seed == 12 and not tem_filter(tail):
            method = os.path.split(fname)[-1].split('+')[0]
            if method == 'itwl':
                label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            if method == 'sad':
                label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            if method == 'z':
                label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            if method == 'tem':
                label = r'TEM' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            fname = tail + 'seed-' + str(seed) + front
        seed = str(seed)
        try:
            base = fname[:-4]

            fname = tail + 'seed-' + seed + front
            #print(fname)
            
            data=np.load(fname)
            if len(data['moves']) != 0:
                results.add_npz(fname)
            else:
                print(f'skipping file {fname} due to no convergence data')
        except BaseException as err:
            raise(err)
            print(f'skipping file {fname}')
            pass
results.median_method(ax, 
                    axins, 
                    subplot=(axs, axins_subplot),
                    moves = 10**np.linspace(6,12,100),
                    E=np.linspace(-system.h_small, 0, 10000),
                    T = np.linspace(system.min_T, 0.25, 1000))
        

plt.figure('latest-entropy')
plt.xlabel(r'$E$')
plt.ylabel(r'$S(E)$')
plt.legend()
plt.ylim(-40,0)
plt.xlim(-1.15,-0.85)
plt.savefig(system.system+'-latest-entropy.svg')
plt.savefig(system.system+'-latest-entropy.pdf')

axs['(c)'].set_xlabel(r'$E$')
axs['(c)'].set_ylabel(r'$S(E)$')

axs['(c)'].set_ylim(-40,0)
axs['(c)'].set_xlim(-1.15,-0.85)

legend = plt.legend(handles = legend_handles(exact=True), loc='lower right')
fig = legend.figure
fig.canvas.draw()

plt.figure('fraction-well')
plt.xlabel(r'E')
plt.ylabel(r'Proportion in Small Well')
plt.legend()
plt.savefig(system.system+'-which.svg')


plt.figure('histogram')
plt.xlabel(r'$E$')
plt.ylabel(r'# of Visitors')
plt.legend()
plt.savefig(system.system+'-histogram.svg')

plt.figure('convergence')
plt.xlabel(r'# of Moves')
plt.ylabel(r'Max Error in $S$')
plt.ylim(1e-3, 1e3)
plt.xlim(1e6, 1e13)
plt.legend()

axs['(a)'].set_xlabel(r'# of Moves')
axs['(a)'].set_ylabel(r'Max Error in $S$')
axs['(a)'].set_ylim(1e-3, 1e3)
axs['(a)'].set_xlim(1e6, 1e13)
ax_legend_2 = plt.gca()
legend = plt.legend(handles=legend_handles())
fig_2 = legend.figure
fig_2.canvas.draw()
#make diagonal lines for convergence
x = np.linspace(1e-30,1e40,2)
y = 1/np.sqrt(x)
for i in range(50):
    plt.loglog(x,y*10**(4*i/5-2), color = 'y',alpha=0.3)
    axs['(a)'].loglog(x,y*10**(4*i/5-2), color = 'y',alpha=0.3)
plt.savefig(system.system+'-convergence.svg')
plt.savefig(system.system+'-convergence.pdf')

plt.figure('convergence-heat-capacity')
plt.xlabel(r'# of Moves')
plt.ylabel(r'Max Error in $C_V$')
plt.ylim(1e-3, 1e3)
plt.xlim(1e6, 1e13)
plt.legend()

axs['(b)'].set_xlabel(r'# of Moves')
axs['(b)'].set_ylabel(r'Max Error in $C_V$')
axs['(c)'].legend(handles = legend_handles(exact=True), loc='lower right')
axs['(b)'].set_ylim(1e-3, 1e3)
axs['(b)'].set_xlim(1e6, 1e13)

#make diagonal lines for convergence
x = np.linspace(1e-10,1e20,2)
y = 1/np.sqrt(x)
for i in range(20):
    plt.loglog(x,y*10**(4*i/5-2), color = 'y',alpha=0.3)
    axs['(b)'].loglog(x,y*10**(4*i/5-2), color = 'y',alpha=0.3)
plt.savefig(system.system+'-heat-capacity-convergence.svg')
plt.savefig(system.system+'-heat-capacity-convergence.pdf')


plt.figure('latest-heat-capacity')
ax.legend(loc='lower right')
axs['(d)'].set_ylabel(r'$C_V$')
axs['(d)'].set_xlabel(r'$T$')
#axs[0,1].legend()

plt.savefig(system.system+'-heat-capacity.svg')
plt.savefig(system.system+'-heat-capacity.pdf')

plt.figure(subplot_fig)
for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.set_title(label, transform=ax.transAxes + trans,
            va='bottom',
            loc='left')
plt.savefig(system.system+'-combined.pdf')

if __name__ == '__main__':
    dump_into_thesis = True
    if dump_into_thesis:
        plt.savefig(os.path.join(r'C:\Users\Henry Sprueill\Documents\Coding\Latex\Thesis\figure', 
                                'step-'+str(step[0]), 
                                system.system+'-combined.pdf'))
    plt.show()