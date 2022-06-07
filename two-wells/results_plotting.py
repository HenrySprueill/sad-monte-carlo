import os
from typing import final
import numpy as np
import glob
import matplotlib.pyplot as plt
import system
import styles_poster as styles
import heat_capacity
from scipy.interpolate import interp1d

class Results:
    def __init__(self, E=None, T=None, S=None, C=None):
        self.fnames = []
        self.mean_e=dict()
        self.mean_which=dict()
        self.hist=dict()
        self.E=dict()
        self.S=dict()
        self.T=dict()
        self.C=dict()
        self.moves=dict()
        self.errors_S=dict()
        self.errors_C=dict()
        self.interp_S=dict()
        self.interp_C=dict()
        self.interp_errors_S=dict()
        self.interp_errors_C=dict()
        self.on_grid_interp_errors_C=dict()

        if E is None or S is None:
            self.exact_S = None
        else:
            self.exact_S = interp1d(E, S, 
                                bounds_error=False,
                                fill_value=np.NaN)
        if T is None or C is None:
            self.exact_C = None
        else:
            self.exact_C = interp1d(T, C, 
                                bounds_error=False,
                                fill_value=np.NaN)
    
    def set_exact_data(self, E, T, S, C):
        self.exact_S = interp1d(E, S, 
                                bounds_error=False,
                                fill_value=np.NaN)
        self.exact_C  = interp1d(T, C, 
                                bounds_error=False,
                                fill_value=np.NaN)
    
    def _query_fname(self, fname,
                            moves=None,
                            E=None,
                            T=None):
        data = dict()
        if fname not in self.fnames:
            return None
        if 'tem+' not in fname:
            data['mean_e']=self.mean_e[fname]
            data['mean_which']=self.mean_which[fname]
            data['hist']=self.hist[fname]
            if E is None:
                data['E']=self.E[fname]
                data['S']=self.S[fname]
            else:
                data['E']=E
                data['S']=self.interp_S[fname](E)
            if moves is None:
                data['errors_S']=self.errors_S[fname]
            else:
                data['errors_S']=self.interp_errors_S[fname](moves)
        if T is None:
            data['T']=self.T[fname]
            data['C']=self.C[fname]
        else:
            data['T']=T
            data['C']=self.interp_C[fname](T)
        if moves is None:
            data['moves']=self.moves[fname]
            data['errors_C']=self.errors_C[fname]
            if fname in self.on_grid_interp_errors_C.keys():
                data['on_grid_errors_C']=self.interp_errors_C[fname](data['moves'])
        else:
            data['moves'] = moves
            data['errors_C']=self.interp_errors_C[fname](moves)
            if fname in self.on_grid_interp_errors_C.keys():
                data['on_grid_errors_C']=self.interp_errors_C[fname](moves)
        return data
    
    def _mean_data(self, fname):
        min_data = np.nan_to_num(self._query_fname(fname), 
                                    nan=np.inf)
        max_data = np.nan_to_num(min_data.copy(), 
                                    nan=-np.inf)
        mean_data = np.nan_to_num(min_data.copy(),
                                    nan=0)

        given_seed = 'seed-'+fname.split('seed-')[-1].split('+')[0]
        #print(given_seed)
        filt = lambda f: \
            f.replace('seed-'+f.split('seed-')[-1].split('+')[0], '') == \
                fname.replace(given_seed, '') and f!=fname
        i=0
        for f in filter(filt, self.fnames):
            #print(f)
            i+=1
            new_data = self._query_fname(f)
            for k in new_data.keys():
                if new_data[k] is not None and k not in ['E','T']:
                    idx = min(len(min_data[k])-1, len(new_data[k])-1)
                    min_data[k] = np.minimum(min_data[k][:idx], np.nan_to_num(new_data[k][:idx], nan=np.inf))
                    max_data[k] = np.maximum(max_data[k][:idx], np.nan_to_num(new_data[k][:idx], nan=-np.inf))

                    mean_data[k] = mean_data[k][:idx] + np.nan_to_num(new_data[k][:idx], nan=0)
        for k in mean_data:
            if mean_data[k] is not None and k not in ['E','T']:
                mean_data[k] = mean_data[k] / i

        return min_data, mean_data, max_data


    def _stack_data_by_key(self, fname, key,
                                moves=None,
                                E=None,
                                T=None):
        data_stack = self._query_fname(fname, moves, E, T)[key]

        given_seed = 'seed-'+fname.split('seed-')[-1].split('+')[0]
        #print(given_seed)
        filt = lambda f: \
            f.replace('seed-'+f.split('seed-')[-1].split('+')[0], '') == \
                fname.replace(given_seed, '') and f!=fname
        for f in filter(filt, self.fnames):
            #print(f)
            new_data = self._query_fname(f, moves, E, T)[key]
            if new_data is not None:
                idx = min(data_stack.shape[-1], len(new_data))
                if len(data_stack.shape) == 1:
                    data_stack = np.vstack((data_stack[:idx], new_data[:idx]))
                else:
                    data_stack = np.vstack((data_stack[:,:idx], new_data[:idx]))
        return data_stack


    def _plot_from_data(self, ax, axins, fname, data=None, data_bounds=None, subplot = None, dump_into_thesis = None):
        if data is None:
            data = self._query_fname(fname)
        if data_bounds is not None:
            lower_data, upper_data = data_bounds
        base = fname[:-4]
        method = os.path.split(fname)[-1].split('+')[0]
        if method == 'itwl':
            label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'sad':
            label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'z':
            label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'tem':
            label = r'TEM' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        

        if method in {'wl','itwl','sad', 'z'}:
            plt.figure('fraction-well')
            plt.plot(data['mean_e'], data['mean_which'], label=label)
        
            if len(data['hist']) != 0:
                plt.figure('histogram')
                plt.plot(data['mean_e'], data['hist'], label=label)

            plt.figure('latest-entropy')
            plt.plot(data['E'][:len(data['S'])], data['S'], 
                                                    label=label, 
                                                    marker = styles.marker(base),
                                                    color = styles.color(base), 
                                                    linestyle= styles.linestyle(base), 
                                                    markevery=25)
            plt.figure('convergence')
            plt.loglog(data['moves'], data['errors_S'], 
                                                label=label, 
                                                marker = styles.marker(base), 
                                                color = styles.color(base), 
                                                linestyle= styles.linestyle(base), 
                                                markevery=2)
            if data_bounds is not None:
                plt.fill_between(lower_data['moves'], lower_data['errors_S'], upper_data['errors_S'],
                                                    color = styles.color(base),
                                                    linestyle=styles.linestyle(base),
                                                    linewidth = 2,
                                                    alpha = 0.2)
            if self.exact_S is not None:
                plt.figure('error_dist_S')
                plt.plot(data['E'], self.exact_S(data['E']) - data['S'], 
                                                label=label, 
                                                marker = styles.marker(base), 
                                                color = styles.color(base), 
                                                linestyle= styles.linestyle(base), 
                                                markevery=400)

            
        
        heat_capacity.plot_from_data(data['T'][:len(data['C'])], data['C'],
                                                                    fname=fname,
                                                                    ax=ax, 
                                                                    axins=axins)

        plt.figure('convergence-heat-capacity')
        plt.loglog(data['moves'], data['errors_C'], 
                                            label=label, 
                                            marker = styles.marker(base), 
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base), 
                                            markevery=2)
        if data_bounds is not None:
            plt.fill_between(lower_data['moves'], lower_data['errors_C'], upper_data['errors_C'],
                                                color = styles.color(base),
                                                alpha = 0.2)
        
        if self.exact_C is not None:
            plt.figure('error_dist_C')
            plt.plot(data['T'], self.exact_C(data['T']) - data['C'], 
                                            label=label,
                                            ms = 5,
                                            marker = styles.marker(base), 
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base), 
                                            markevery=400)
        
        if subplot is not None:
            axs = subplot[0]
            axins_subplot = subplot[1]
            if method in {'wl','itwl','sad','z'}:
                axs['(c)'].plot(data['E'][:len(data['S'])], data['S'], 
                                                        label=label, 
                                                        marker = styles.marker(base),
                                                        color = styles.color(base), 
                                                        linestyle= styles.linestyle(base), 
                                                        markevery=250)
            elif method == 'z':
                axs['(c)'].plot(data['E'], data['S'], 
                                        label=label, 
                                        color = styles.color(base), 
                                        linestyle= styles.linestyle(base))
            
            heat_capacity.plot_from_data(data['T'][:len(data['C'])], data['C'],
                                                                        fname=fname,
                                                                        ax=axs['(d)'], 
                                                                        axins=axins_subplot)

            plt.figure('convergence')
            if method in {'wl','itwl','sad','z'}:
                axs['(a)'].loglog(data['moves'], data['errors_S'], 
                                                    label=label, 
                                                    marker = styles.marker(base), 
                                                    color = styles.color(base), 
                                                    linestyle= styles.linestyle(base), 
                                                    markevery=2)
                if data_bounds is not None:
                    axs['(a)'].fill_between(lower_data['moves'], lower_data['errors_S'], upper_data['errors_S'],
                                                        color = styles.color(base),
                                                        linestyle=styles.linestyle(base),
                                                        linewidth = 2,
                                                        alpha = 0.2)


            plt.figure('convergence-heat-capacity')
            axs['(b)'].loglog(data['moves'], data['errors_C'], 
                                                label=label, 
                                                marker = styles.marker(base), 
                                                color = styles.color(base), 
                                                linestyle= styles.linestyle(base), 
                                                markevery=2)

            if data_bounds is not None:
                axs['(b)'].fill_between(lower_data['moves'], lower_data['errors_C'], upper_data['errors_C'],
                                                    color = styles.color(base),
                                                    alpha = 0.2)
        


    def add_npz(self, fname):
        if fname[-4:] != '.npz':
            raise('Incorrect filetype')
        self.fnames.append(fname)
        seed = fname.split('seed-')[-1].split('+')[0]
        data = np.load(fname)
        
        self.T[fname] = data['T']
        self.moves[fname] = data['moves']
        if 'tem+' not in fname:
            self.mean_e[fname] = data['mean_e']
            self.mean_which[fname] = data['mean_which']
            try:
                self.hist[fname] = data['hist']
            except:
                self.hist[fname] = None
            self.S[fname] = data['S']
            self.E[fname] = data['E']
            self.errors_S[fname] = data['errors_S']
            self.interp_S[fname]=interp1d(data['E'], data['S'], 
                                bounds_error=False,
                                fill_value=np.NaN)
            self.interp_errors_S[fname]=interp1d(data['moves'], data['errors_S'], 
                                bounds_error=False,
                                fill_value=np.NaN)
        self.C[fname] = data['C']
        self.errors_C[fname] = data['errors_C']
        self.interp_C[fname]=interp1d(data['T'], data['C'], 
                                bounds_error=False,
                                fill_value=np.NaN)
        self.interp_errors_C[fname]=interp1d(data['moves'], data['errors_C'], 
                                bounds_error=False,
                                fill_value=np.NaN)
        if 'on_grid_errors_C' in data.keys():
            self.on_grid_interp_errors_C[fname]=interp1d(data['moves'], data['on_grid_errors_C'], 
                                bounds_error=False,
                                fill_value=np.NaN)


    def plot_seed(self,
                    ax,
                    axins,
                    seed, 
                    method = None, 
                    additional_filters = None):
        seed = str(seed)
        filter_seed_hist = lambda f: 'seed-'+seed+'+' in f
        filter_seed_replicas = lambda f: 'seed-'+seed+'+' in f
        filters = [filter_seed_hist, filter_seed_replicas]
        if method is not None:
            method_filter = lambda f: method in f
        else:
            method_filter = lambda f: True
        filters.append(method_filter)
        if additional_filters is None:
            additional_filters = lambda f: True
        filters.append(additional_filters)
        fnames = self.fnames
        for filt in filters:
            fnames = filter(filt, fnames)

        for f in fnames:
            self._plot_from_data(ax, axins, f)
            

    def mean_method(self, ax, axins, subplot = None, dump_into_thesis = None):
        stacked_data = dict()
        unstacked_data = dict()
        for f in filter(lambda f: 'seed-1+' in f, self.fnames):
            for k in self._query_fname(f).keys():
                if 'error' in k or 'moves' in k: 
                    stacked_data[k] = self._stack_data_by_key(f, k)
                else:
                    unstacked_data[k] = self._query_fname(f)[k]
            min_data = dict()
            max_data = dict()
            for k in stacked_data.keys():
                min_data[k] = np.nanmin(stacked_data[k], axis=0)
                unstacked_data[k] = np.nanmean(stacked_data[k], axis=0)
                max_data[k] = np.nanmax(stacked_data[k], axis=0)
            #print(max_data['errors_S'])
            self._plot_from_data(ax, axins, f, 
                                    data = unstacked_data, 
                                    data_bounds=(min_data, max_data), 
                                    subplot = subplot, 
                                    dump_into_thesis = dump_into_thesis)
    
    def median_method(self, ax, axins, 
                                subplot = None, 
                                dump_into_thesis = None,
                                moves=None,
                                E=None,
                                T=None):
        stacked_data = dict()
        unstacked_data = dict()
        for f in filter(lambda f: 'seed-1+' in f, self.fnames):
            for k in self._query_fname(f).keys():
                if 'error' in k or 'moves' in k: 
                    stacked_data[k] = self._stack_data_by_key(f, k,
                                                        moves=moves,
                                                        E=E,
                                                        T=T)
                else:
                    unstacked_data[k] = self._query_fname(f,
                                                        moves=moves,
                                                        E=E,
                                                        T=T)[k]
            min_data = dict()
            max_data = dict()
            for k in stacked_data.keys():
                min_data[k] = np.nanmin(stacked_data[k], axis=0)
                unstacked_data[k] = np.nanmedian(stacked_data[k], axis=0)
                max_data[k] = np.nanmax(stacked_data[k], axis=0)
            #print(max_data['errors_S'])
            self._plot_from_data(ax, axins, f, 
                                    data = unstacked_data, 
                                    data_bounds=None,#(min_data, max_data), 
                                    subplot = subplot, 
                                    dump_into_thesis = dump_into_thesis)

    def convergence_method(self, axs,
                    moves = None,
                    E= None,
                    T = None):
        stacked_data = dict()
        unstacked_data = dict()
        for f in filter(lambda f: 'seed-1+' in f, self.fnames):
            for k in self._query_fname(f).keys():
                if 'error' in k or 'moves' in k: 
                    stacked_data[k] = self._stack_data_by_key(f, k,
                                                        moves=moves,
                                                        E=E,
                                                        T=T)
                else:
                    unstacked_data[k] = self._query_fname(f,
                                                        moves=moves,
                                                        E=E,
                                                        T=T)[k]
            min_data = dict()
            max_data = dict()
            for k in stacked_data.keys():
                min_data[k] = np.nanmin(stacked_data[k], axis=0)
                unstacked_data[k] = np.nanmedian(stacked_data[k], axis=0)
                max_data[k] = np.nanmax(stacked_data[k], axis=0)
            
            if any([method in f for method in ['sad', 'itwl']]):
                if 'step-0.01' in f:
                    S_ax = axs['(a)'] if '(a)' in axs.keys() else None
                    C_ax = axs['(b)'] if '(b)' in axs.keys() else None
                    self._plot_histogram_convergence(S_ax, C_ax, f, data=unstacked_data)
                elif 'step-0.001' in f:
                    self._plot_histogram_convergence(axs['(c)'], axs['(d)'], f, data=unstacked_data)
                elif 'step-0.0001' in f:
                    S_ax = axs['(e)'] if '(e)' in axs.keys() else None
                    C_ax = axs['(f)'] if '(f)' in axs.keys() else None
                    self._plot_histogram_convergence(S_ax, C_ax, f, data=unstacked_data)
            if any([method in f for method in ['z', 'tem']]):
                if 'tem' in f:
                    S_ax = None
                    C_ax = axs['(b)'] if '(b)' in axs.keys() else None
                    self._plot_replica_convergence(None, C_ax, f, data=unstacked_data)
                elif 'z+' in f:
                    S_ax = axs['(a)'] if '(a)' in axs.keys() else None
                    C_ax = axs['(b)'] if '(b)' in axs.keys() else None
                    self._plot_replica_convergence(S_ax, C_ax, f, data=unstacked_data)
    

    def _plot_histogram_convergence(self, S_ax, C_ax, fname, data=None):

        if data is None:
            data = self._query_fname(fname)
        base = fname[:-4]
        method = os.path.split(fname)[-1].split('+')[0]
        if method == 'itwl':
            label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'sad':
            label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'z':
            return
            label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'tem':
            return
            label = r'TEM' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if S_ax is not None:
            S_ax.loglog(data['moves'], data['errors_S'], 
                                                label=label, 
                                                marker = styles.marker(base),
                                                ms=16,
                                                color = styles.color(base), 
                                                linestyle= '-', 
                                                markevery=4)
        if C_ax is not None:
            C_ax.loglog(data['moves'], data['errors_C'], 
                                                label=label, 
                                                marker = styles.marker(base), 
                                                color = styles.color(base), 
                                                linestyle= '-', 
                                                markevery=4)

    def _plot_replica_convergence(self, S_ax, C_ax, fname, data=None):
        if data is None:
            data = self._query_fname(fname)
        base = fname[:-4]
        method = os.path.split(fname)[-1].split('+')[0]
        if method == 'itwl':
            label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'sad':
            label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'z':
            label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'tem':
            label = r'PT' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if S_ax is not None:
            S_ax.loglog(data['moves'], data['errors_S'], 
                                                label=label, 
                                                marker = styles.marker(base), 
                                                color = styles.color(base), 
                                                ms=16,
                                                linestyle= '-', 
                                                markevery=4)
        on_grid = False
        if C_ax is not None:
            if 'on_grid_errors_C' not in data.keys() or not on_grid:
                C_ax.loglog(data['moves'], data['errors_C'], 
                                                    label=label, 
                                                    marker = styles.marker(base), 
                                                    color = styles.color(base),
                                                    ms=16, 
                                                    linestyle='-', 
                                                    markevery=4)
            else:
                C_ax.loglog(data['moves'], data['on_grid_errors_C'], 
                                                    label=label, 
                                                    marker = styles.marker(base), 
                                                    color = styles.color(base), 
                                                    linestyle= '-', 
                                                    markevery=4)

    def plot_tempering_distribution(self, fname, dist_ax, C_ax, T):

        data = self._query_fname(fname, T=T)
        data_uninterp = self._query_fname(fname)
        base = fname[:-4]
        method = os.path.split(fname)[-1].split('+')[0]
        if method == 'itwl':
            label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            return
        if method == 'sad':
            label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            return
        if method == 'z':
            label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            return
        if method == 'tem':
            label = r'PT' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        dist_ax.plot(T, -(self.exact_C(T) - data['C']), 
                                            label=label,
                                            ms = 5,
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base), 
                                            markevery=400)
        dist_ax.scatter(data_uninterp['T'], -(self.exact_C(data_uninterp['T']) - data_uninterp['C']), 
                                            label=label,
                                            marker = styles.marker(base),
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base))

        C_ax.plot(T, data['C'], 
                                            label=label,
                                            ms = 5,
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base), 
                                            markevery=400)
        C_ax.scatter(data_uninterp['T'], data_uninterp['C'], 
                                            label=label,
                                            marker = styles.marker(base),
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base))
        
        C_ax.plot(T, self.exact_C(T), 
                                            label='exact',
                                            marker = styles.marker(None), 
                                            color = 'k', 
                                            linestyle= ':')

                        
    
    def plot_entropy_distribution(self, fname, dist_ax, S_ax, E):
        data = self._query_fname(fname, E=E)
        base = fname[:-4]
        if styles.get_barrier(base) == '1e-1':
            return
        method = os.path.split(fname)[-1].split('+')[0]
        if method == 'itwl':
            label = r'$1/t$-WL' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if 'sad' in fname:
            label = r'SAD' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            return
        if method == 'z':
            label = r'ZMC' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
        if method == 'tem':
            label = r'PT' + r'-$E_{barr}$=0.'+styles.get_barrier(base)[0]
            return
        dist_ax.plot(E, data['S'] - self.exact_S(E), 
                                            label=label,
                                            marker = styles.marker(base),
                                            color = styles.color(base), 
                                            linestyle= '-', 
                                            markevery=1000)

        S_ax.plot(E, data['S'], 
                                            label=label,
                                            marker = styles.marker(base),
                                            color = styles.color(base), 
                                            linestyle= styles.linestyle(base), 
                                            markevery=1000)
        
        S_ax.plot(E, self.exact_S(E), 
                                            label='exact',
                                            marker = styles.marker(None), 
                                            color = 'k', 
                                            linestyle= ':')