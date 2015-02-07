from neo.io import AxonIO
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter 
from scipy import signal
from scipy.stats import circmean, circstd
from plotting_help import *
import sys, os
import scipy.signal
from bisect import bisect
import cPickle
import math
import pandas as pd
import scipy as sp

#---------------------------------------------------------------------------#

class Phys_Flight():  
    def __init__(self, fname):
        if fname.endswith('.abf'):
            self.basename = ''.join(fname.split('.')[:-1])
            self.fname = fname
        else:
            self.basename = fname
            self.fname = self.basename + '.abf'  #check here for fname type 
                  
    def open_abf(self,exclude_indicies=[]):        
        abf = read_abf(self.fname)              
        
        # added features to exclude specific time intervals
        n_indicies = np.size(abf['x_ch']) #assume all channels have the same sample #s 
        inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
                   
        self.xstim = np.array(abf['x_ch'])[inc_indicies]
        self.ystim = np.array(abf['y_ch'])[inc_indicies]

        self.samples = np.arange(self.xstim.size)  #this is adjusted
        self.t = self.samples/float(10000) # sampled at 10,000 hz -- encode here?
 
        #now process wing signal
        lwa_v = np.array(abf['wba_l'])[inc_indicies]
        rwa_v = np.array(abf['wba_r'])[inc_indicies]
        
        self.lwa = process_wings(lwa_v)
        self.rwa = process_wings(rwa_v)
        
        lmr = self.lwa - self.rwa
        self.raw_lmr = lmr
        self.lmr = lmr
        
        self.ao = np.array(abf['patid'])[inc_indicies]
        
        self.vm = np.array(abf['vm'])[inc_indicies] - 13 #offset for bridge potential
        self.tach = np.array(abf['tach'])[inc_indicies]
        self.n_nonflight_trs = 0 #will update
            
    def _is_flying(self, start_i, stop_i, percent_thres = .90):  #fix this critera
        #check that animal is flying using the tachometer signal
     
        #iterate through the trace in steps of 25 ms (250 with typical 10,000 sampling rate)
        #min flight rate of interest = 100 wing beats/s
        tach_range_thres = 1.5 #2
        stroke_range_thres = 1
        step_size = 350 #250
        i_steps = range(start_i,stop_i,step_size)
        n_tests = np.size(i_steps)
        flight_tests = np.ones(n_tests, dtype=bool)    
        
        #check tachometer, but also make sure flight track is on. 
        for interval_start,test_i in zip(i_steps,range(n_tests)):
            interval = np.arange(interval_start,(interval_start+step_size))
            interval_min = np.min(self.tach[interval])
            interval_max = np.max(self.tach[interval])
            stroke_min = np.min(self.lmr[interval])
            stroke_max = np.max(self.lmr[interval])
            
            tach_flight = (interval_max - interval_min) > tach_range_thres
            stroke_flight = (stroke_max - stroke_min) > stroke_range_thres
            
            flight_tests[test_i] = tach_flight and stroke_flight
        
        is_flying = np.nanmean(flight_tests) > percent_thres 
        return is_flying
        
           
#---------------------------------------------------------------------------#

class Spot_Phys(Phys_Flight):
    
    def process_fly(self,ex_i=[]):  #does this interfere with the Flight_Phys init?
        self.open_abf(ex_i)
        self.clean_lmr_signal()
        self.parse_trial_times()
        self.parse_stim_type()
        
    def show_nonflight_exclusion(self,title_txt=''):
        fig = plt.figure(figsize=(17.5,4.5))
        plt.title(title_txt)
        
        #plt.plot(self.tach*2-75,color=purple)
        plt.plot(self.raw_lmr[::10],color=blue)
        plt.plot(self.lmr,color=magenta)
        plt.plot(self.ao-100,color=black)
        plt.plot(self.tr_starts,np.ones_like(self.tr_starts),'oc')
        plt.ylabel('WBA (Degrees)')
        plt.xlabel('Samples (at 10,000 hz)')
        
        blue_line = mlines.Line2D([], [], color='blue',label='Raw lmr')
        magenta_line = mlines.Line2D([], [], color='magenta',label='Interpolated lmr')
        purple_line = mlines.Line2D([], [], color='purple',label='Tachometer')
        black_line = mlines.Line2D([], [], color='black',label='AO')
        cyan_pt = mlines.Line2D([],[],marker='o',linewidth=0, color='cyan',label='Tr starts')
        
        fontP = FontProperties()
        fontP.set_size('small')
        
        plt.legend(handles=[blue_line,magenta_line,purple_line,black_line,cyan_pt], prop = fontP, \
                            bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
        
        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_nonflight exclusion.png',bbox_inches='tight',dpi=100) 
        
    def clean_lmr_signal(self,title_txt='',if_plot=False):
        lmr = np.copy(self.lmr)
        cleaned_lmr = np.copy(lmr) # make a copy here
        
        d_lmr = np.diff(lmr)
        artifacts = np.where(abs(d_lmr) > 35)[0]-1
    
        if np.size(artifacts) >= 1:
            artifact_gap_start_is = np.where(np.diff(artifacts) > .1*10000)[0]+1
            artifact_gap_start_is = np.hstack((0,artifact_gap_start_is))  # add first start
    
            artifact_gap_stop_is = artifact_gap_start_is[1:]-1
            artifact_gap_stop_is = np.hstack((artifact_gap_stop_is,np.size(artifacts)-1)) # add last

            # now loop though all of these blanked periods, fill with previous/last real value
            for start_i, stop_i in zip(artifact_gap_start_is,artifact_gap_stop_is):
                fill_i = artifacts[start_i] - 1
                if fill_i < 0: 
                    fill_i = lmr_stop_i[stop_i] + 1
            
                lmr_start_i = artifacts[start_i]
                lmr_stop_i = artifacts[stop_i] + 10
        
                cleaned_lmr[lmr_start_i:lmr_stop_i] = cleaned_lmr[fill_i]
            
        if if_plot:
            fig = plt.figure()
            plt.plot(lmr,color=blue)
            
            if artifacts:
                plt.plot(artifacts,lmr[artifacts],'*c')
                plt.plot(artifacts[artifact_gap_start_is],np.ones_like(artifact_gap_start_is),'og')
                plt.plot(artifacts[artifact_gap_stop_is]+10,np.ones_like(artifact_gap_stop_is),'om')
            plt.plot(cleaned_lmr,color=purple)
            plt.title(title_txt)
            
            # also show nonflight periods here

        self.lmr = cleaned_lmr
          
    def remove_non_flight_trs(self, iti=750):
        # loop through each trial and determine whether fly was flying continuously
        # if a short nonflight bout (but not during turn window), interpolate
        #
        # delete the trials with long nonflight bouts--change n_trs, tr_starts, 
        # tr_stops, looming stim on
        
        non_flight_trs = [];
        
        for tr_i in range(self.n_trs):
            this_tr_start = self.tr_starts[tr_i] - iti
            this_tr_stop = self.tr_stops[tr_i] + iti
            
            if not self._is_flying(this_tr_start,this_tr_stop, percent_thres = .90):
                non_flight_trs.append(tr_i) 
        
        #print 'nonflight trials : ' + ', '.join(str(x) for x in non_flight_trs)
        
        print 'nonflight trials : ' + str(np.size(non_flight_trs)) + '/' + str(self.n_trs)
        
        #now remove these
        self.n_nonflight_trs = np.size(non_flight_trs)
        self.n_trs = self.n_trs - np.size(non_flight_trs)
        self.tr_starts = np.delete(self.tr_starts,non_flight_trs)  #index values of starting and stopping
        self.tr_stops = np.delete(self.tr_stops,non_flight_trs)
        #self.pre_loom_stim_ons = np.delete(self.pre_loom_stim_ons,non_flight_trs)
                    
    def parse_trial_times(self, if_debug_fig=False):
        # parse the ao signal to determine trial start and stop index values
        # include checks for unusual starting aos, early trial ends, 
        # long itis, etc
        #
        # for this protocol, use the ao for coarse alignment
        # and use the xstim for exact timing
        
        
        ao_diff = np.diff(self.ao)
        
        tr_start = self.samples[np.where(ao_diff <= -5)]
        start_diff = np.diff(tr_start)
        redundant_starts = tr_start[np.where(start_diff < 1000)]
        clean_tr_starts_unaligned = np.setdiff1d(tr_start,redundant_starts)+1
        clean_tr_starts = clean_tr_starts_unaligned
        
        # now shift tr_start based on the xstim signal
        # look within a ~3000 window for the start
        clean_tr_starts = clean_tr_starts - 1500  # ***** update this to be exact. later
        
        # max_shift = int(3500e6)
#         n_starts = np.size(clean_tr_starts_unaligned)
#         clean_tr_starts = np.ones_like(clean_tr_starts_unaligned)
#         
#         for this_start,tr_i in zip(clean_tr_starts_unaligned[0:2],range(2)):
#             this_i_range = range((this_start-max_shift),this_start)
#             lower_bound_is = np.where(self.xstim[this_i_range] > .25)[0]
#             upper_bound_is = np.where(self.xstim[this_i_range] < .35)[0]
#             in_window_is = np.intersect1d(lower_bound_is,upper_bound_is)
#             
#             clean_tr_starts[tr_i] = this_i_range[in_window_is[0]]
#             
#         
        tr_stop = self.samples[np.where(ao_diff >= 5)]
        stop_diff = np.diff(tr_stop)
        redundant_stops = tr_stop[np.where(stop_diff < 1000)] 
        clean_tr_stops = np.setdiff1d(tr_stop,redundant_stops)+1
        
        # check that first start is before first stop
        if clean_tr_stops[0] < clean_tr_starts[0]: 
            clean_tr_stops = np.delete(clean_tr_stops,0)
         
        # last stop is after last start
        if clean_tr_starts[-1] > clean_tr_stops[-1]:
            clean_tr_starts = np.delete(clean_tr_starts,len(clean_tr_starts)-1)
            
        # check for two starts in a row
        if clean_tr_starts[1] < clean_tr_stops[0]:    
            clean_tr_starts = np.delete(clean_tr_starts,0)
        
        # now overwrite clean_tr_stops
        clean_tr_stops = clean_tr_starts + 9080
        
        # define tr_stop based on the end of xstim plateauing
        # around 8.6 ao
        # alternatively, I could just set a fixed time for now.
        
        # should check for same # of starts and stops *****
        # add a warning here if these don't align
        n_trs = len(clean_tr_starts)
        
        print np.size(clean_tr_starts), np.size(clean_tr_stops)
        
        if if_debug_fig:
            figd = plt.figure()
            plt.plot(self.ao)
            plt.plot(ao_diff,color=magenta)
            
            plt.plot(self.xstim,'green')
            y_start = np.ones(len(clean_tr_starts))
            y_stop = np.ones(len(clean_tr_stops))
            plt.plot(clean_tr_starts,y_start*7,'go')
            plt.plot(clean_tr_stops,y_stop*7,'ro')
            
        
        #detect when the y stim stepped
        ystim_diff = np.diff(self.ystim)
        y_step = self.samples[np.where(ystim_diff > .03)]

        ##now discriminate first stim on for a trial, not looming steps
        #pre_loom_stim = np.zeros(n_trs)
        #for tr_i in range(n_trs):
        #    earliest_t = clean_tr_starts[tr_i] - 2000  #look within a window before looming
        #    latest_t = clean_tr_starts[tr_i] - 300 
        #    win1 = np.where(y_step > earliest_t)[0]
        #    win2 = np.where(y_step < latest_t)[0]
        #    candidate_stim_starts = y_step[np.intersect1d(win1,win2)]
        #    pre_loom_stim[tr_i] = candidate_stim_starts[0]
        
        #next encode post loom stim on, iti_dur as the median
        
        self.n_trs = n_trs 
        self.tr_starts = clean_tr_starts  #index values of starting and stopping
        self.tr_stops = clean_tr_stops
        #self.pre_loom_stim_ons = pre_loom_stim
        
        #here remove all trials in which the fly is not flying. 
        #self.remove_non_flight_trs()
        
    def parse_stim_type(self):
        #calculate the stimulus type
        
        # 3.80; % [1] Spot on L, moving back to front (4th stimulus, in order)
        # 3.90; % [2] Spot on L, moving front to back (3rd stimulus, in order)
        # 2.50; % [3] Spot on R, moving back to front (2nd stimulus, in order)
        # 2.40; % [4] Spot on R, moving front to back (1st stimulus, in order)
        
        stim_types_labels = np.asarray(['Spot on right, moving front to back',\
                                        'Spot on right, moving back to front',\
                                        'Spot on left, moving front to back',\
                                        'Spot on left, moving back to front'])
        
        stim_types = -1*np.ones(self.n_trs,'int')
        tr_ao_codes = np.empty(self.n_trs)
        
        #first loop through to get the unique ao values
        for tr in range(self.n_trs): 
            this_start = self.tr_starts[tr]-4000
            this_stop = self.tr_starts[tr]
            tr_ao_codes[tr] = round(np.mean(self.ao[this_start:this_stop]),1)   
        unique_tr_ao_codes = np.unique(tr_ao_codes) 
        print 'trial types = ' + str(unique_tr_ao_codes)
        
        for tr in range(self.n_trs): 
            tr_ao_code = tr_ao_codes[tr]         
            
            if not np.isnan(tr_ao_code):
                stim_types[tr] = int(np.where(unique_tr_ao_codes == tr_ao_code)[0])
                
        
        self.stim_types = stim_types  #change to integer, although nans are also useful
        self.stim_types_labels = stim_types_labels
           
    def plot_vm_wba_stim_corr(self,title_txt='',vm_base_subtract=False,\
                              vm_lim=[-80,-50],wba_lim=[-45,45],if_save=False,if_x_zoom=True): 
    
        # make figure four rows of signals -- vm, wba, stimulus, vm-wba corr x
        # four columns of stimulus types
        
        sampling_rate = 10000 # in hertz
        s_iti = 1 * sampling_rate  # ****************** not sure what this is? 
        baseline_win = range(0,int(.5*sampling_rate))  
                # time relative spot movement start - s_iti
                # do not average out the visual onset
        
        #get all traces and detect saccades ______________________________________________
        
        # this step is very slow. for debugging, run this once, pickle, then load
        # all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)
        
        all_fly_traces = pd.read_pickle('all_fly_traces.save')
        
        
        
        fig = plt.figure(figsize=(12,8))  
        gs = gridspec.GridSpec(4,4,height_ratios=[1,.75,.1,.25])
        gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    
        #store all subplots for formatting later           
        all_vm_ax = []
        all_wba_ax = []
        all_stim_ax = []
        all_corr_ax = []
    
        cnds_to_plot = range(4)
        
        # now loop through the conditions/columns. ____________________________________
        # the signal types are encoded in separate rows(vm, wba, stim, corr)
        for cnd, grid_col in zip(cnds_to_plot,range(4)):
        
            this_cnd_trs = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
            n_cnd_trs = 15 #np.size(this_cnd_trs)
            
            # get colormap info ______________________________________________________
            cmap = plt.cm.get_cmap('jet') #get a better colormap.m jet's luminance changes are confusing. **********
            cNorm  = colors.Normalize(0,n_cnd_trs)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
            # create subplots ________________________________________________________              
            if grid_col == 0:
                vm_ax = plt.subplot(gs[0,grid_col])
                wba_ax = plt.subplot(gs[1,grid_col], sharex=vm_ax) 
                stim_ax = plt.subplot(gs[2,grid_col],sharex=vm_ax)    
                corr_ax = plt.subplot(gs[3,grid_col],sharex=vm_ax)        
            else:
                vm_ax = plt.subplot(gs[0,grid_col],  sharey=all_vm_ax[0])
                wba_ax = plt.subplot(gs[1,grid_col], sharex=vm_ax,sharey=all_wba_ax[0]) 
                stim_ax = plt.subplot(gs[2,grid_col],sharex=vm_ax,sharey=all_stim_ax[0])    
                corr_ax = plt.subplot(gs[3,grid_col],sharex=vm_ax,sharey=all_corr_ax[0])
            all_vm_ax.append(vm_ax)   #can I preallocate the size here? data types? *****************
            all_wba_ax.append(wba_ax) 
            all_stim_ax.append(stim_ax)
            all_corr_ax.append(corr_ax)
        
            # loop single trials and plot all signals ________________________________
            for tr, i in zip(this_cnd_trs[0:n_cnd_trs],range(n_cnd_trs)):
           
                this_color = scalarMap.to_rgba(i)        
                
                # plot Vm signal _____________________________________________________      
                vm_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'vm')] 
                
                if vm_base_subtract:
                    vm_base = np.nanmean(vm_trace[baseline_win])
                    vm_trace = vm_trace - vm_base
                
                non_nan_i = np.where(~np.isnan(vm_trace))[0]  #I shouldn't need these. remove nans earlier. ****************
                vm_ax.plot(vm_trace[non_nan_i],color=this_color)
                
                #filtered_vm_trace = butter_lowpass_filter(vm_trace[non_nan_i],10)
                #vm_ax.plot(filtered_vm_trace,color=this_color)
                
                # plot WBA signal ____________________________________________________           
                wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
            
                baseline = np.nanmean(wba_trace[baseline_win])
                wba_trace = wba_trace - baseline  
                 
                non_nan_i = np.where(~np.isnan(wba_trace))[0]  ##remove nans earlier/check to make sure nans only occur at the end
                filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],6)
                wba_ax.plot(filtered_wba_trace,color=this_color)
          
                #now plot stimulus traces ____________________________________________
                stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'xstim')],color=this_color)
                              
            
        # now format all subplots _____________________________________________________  
       
        vm_lim = vm_ax.get_ylim()
        wba_lim = wba_ax.get_ylim()
        
        # loop though all columns again, format each row ______________________________
        for col, cnd in zip(range(4),cnds_to_plot):      
            #create shaded regions of baseline vm and saccade time ___________________
            vm_min_t = baseline_win[0]
            vm_max_t = baseline_win[-1]
            all_vm_ax[col].fill([vm_min_t,vm_max_t,vm_max_t,vm_min_t],[vm_lim[1],vm_lim[1],vm_lim[0],vm_lim[0]],'black',alpha=.1)
                     
            # set the ylim for the stimulus and correlation rows ______________________
            all_stim_ax[col].set_ylim([0,10])
            all_corr_ax[col].set_ylim([-1,1])
             
            # label axes, show xlim and ylim __________________________________________
            
            # remove all time xticklabels
            all_vm_ax[col].tick_params(labelbottom='off')
            all_wba_ax[col].tick_params(labelbottom='off')
            all_stim_ax[col].tick_params(labelbottom='off')
            all_corr_ax[col].tick_params(labelbottom='off')
            
            #if if_x_zoom:
            #    all_vm_ax[col].set_xlim()
            #else:
            #    all_vm_ax[col].set_xlim([0,max_t])
            
            
            all_vm_ax[col].relim()
            all_vm_ax[col].autoscale_view(True,True,True)
                
            
            if col == 0: #label yaxes
            
                if vm_base_subtract:
                    all_vm_ax[col].set_ylabel('Baseline subtracted Vm (mV)')
                else:
                    all_vm_ax[col].set_ylabel('Vm (mV)')
                all_wba_ax[col].set_ylabel('WBA (V)')
                #all_stim_ax[col].set_ylabel('Stim (frame)')
                all_stim_ax[col].set_yticks([])
                
                all_corr_ax[col].set_ylabel('Corr(Vm, WBA)')
                
                vm_ax_ylim = all_vm_ax[col].get_ylim()
                all_vm_ax[col].set_yticks([vm_ax_ylim[0],0,vm_ax_ylim[1]])
                
                wba_ax_ylim = all_wba_ax[col].get_ylim()
                all_wba_ax[col].set_yticks([wba_ax_ylim[0],0,wba_ax_ylim[1]])
                
                corr_ax_ylim = all_corr_ax[col].get_ylim()
                all_corr_ax[col].set_yticks([corr_ax_ylim[0],0,corr_ax_ylim[1]])
                
                # label time x axis for just col 0 ______________________
                # divide by sampling rate _______________________________
                def div_sample_rate(x, pos): 
                    #The two args are the value and tick position 
                    return (x-s_iti)/sampling_rate
                    
                formatter = FuncFormatter(div_sample_rate) 
                all_corr_ax[col].xaxis.set_major_formatter(formatter)
                                    
                all_corr_ax[col].tick_params(labelbottom='on')
                all_corr_ax[col].set_xlabel('Time from loom start (s)') 

            else: # remove all ylabels 
                all_vm_ax[col].tick_params(labelleft='off')
                all_wba_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelleft='off')
                all_corr_ax[col].tick_params(labelleft='off')
              
        #now annotate stimulus positions, title ______________________________________      
        #fig.text(.22,.905,'Left',fontsize=14)
        #fig.text(.775,.905,'Right',fontsize=14)
        
        figure_txt = title_txt
        fig.text(.425,.95,figure_txt,fontsize=18) 
        
        #fig.text(.05,.95,tr_info_str,fontsize=14) 
               
        plt.draw()
        
        if if_save:
            saveas_path = '/Users/jamie/bin/figures/'
            if if_x_zoom:
                plt.savefig(saveas_path + figure_txt + '_fast_spot_vm_wings_corr_zoomed.png',\
                bbox_inches='tight',dpi=100) 
            else:
                plt.savefig(saveas_path + figure_txt + '_fast_spot_vm_wings_corr.png',\
                bbox_inches='tight',dpi=100) 
            plt.close('all')
             
    def plot_each_tr_saccade(self,l_div_v_list=[0],
        wba_lim=[-45,45]): 
        #for each l/v stim parameter, 
        #make figure four rows of signals -- vm, wba, stimulus, vm-wba corr x
        #three columns of looming direction
        
        #time windows in which to examine turning behaviors. these are by eye
        sampling_rate = 10000
        l_div_v_turn_windows = []
        l_div_v_turn_windows.append(range(int(2.45*sampling_rate),int(2.8*sampling_rate)))
        l_div_v_turn_windows.append(range(int(2.95*sampling_rate),int(3.3*sampling_rate)))
        l_div_v_turn_windows.append(range(int(3.85*sampling_rate),int(4.20*sampling_rate)))
        
        s_iti = 20000   #add iti periods
        baseline_win = range(0,5000)  #be careful not to average out the visual transient here.
        
        #get all traces __________________________________________________________________
        all_fly_traces = self.get_traces_by_stim('this_fly',s_iti) 
        
        #now plot one figure for each looming speed ______________________________________
        for loom_speed in l_div_v_list: 
        
            cnds_to_plot = np.arange(0,7,3) + loom_speed
            #0 1 2 ; 3 4 5 ; 6 7 8 
            this_turn_win = l_div_v_turn_windows[loom_speed]
            
            #now loop through the conditions/columns. ____________________________________
            #the signal types are encoded in separate rows(vm, wba, stim, corr)
            for cnd in cnds_to_plot[1:3]:
            
                this_cnd_trs = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
                n_cnd_trs = np.size(this_cnd_trs)
            
                #loop single trials and plot all signals _________________________________
                for tr in this_cnd_trs:
                    
                    #plot Vm signal ______________________________________________________      
                    #vm_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'vm')]
                    
                    #plot WBA signal _____________________________________________________           
                    wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
                    baseline = np.nanmean(wba_trace[baseline_win])
                    wba_trace = wba_trace - baseline  #always subtract the baseline here
                    
                    saccade_time = find_saccades(wba_trace,True)
                    plt.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'ystim')])
    
    
                   
    def get_traces_by_stim(self,fly_name='this_fly',iti=25000,get_saccades=False):
    #here extract the traces for each of the stimulus times. 
    #align to looming start, and add the first pre stim and post stim intervals
    #here return a data frame of lwa and rwa wing traces
    #self.stim_types already holds an np.array vector of the trial type indicies
   
    #using a pandas data frame with multilevel indexing! rows = time in ms
    #columns are multileveled -- genotype, fly, trial index, trial type, trace
        
        pre_loom_stim_dur = 10000 #add this to the flies? 
        
        fly_df = pd.DataFrame()
        fly_saccades_df = pd.DataFrame() #keep empty if not tracking all saccades
       
        for tr in range(self.n_trs):
            this_loom_start = self.tr_starts[tr]
            this_start = this_loom_start - iti
            this_stop = self.tr_stops[tr] + pre_loom_stim_dur
            
            this_stim_type = self.stim_types[tr]
            iterables = [[fly_name],
                         [tr],
                         [this_stim_type],
                         ['lmr','vm','xstim']]
            column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type','trace']) 
                                                            #is the unsorted tr_type level a problem?    
            
            
            tr_traces = np.asarray([self.lmr[this_start:this_stop],
                                    self.vm[this_start:this_stop],
                                    self.xstim[this_start:this_stop]]).transpose()  #reshape to avoid transposing
                                      
            tr_df = pd.DataFrame(tr_traces,columns=column_labels) #,index=time_points) 
            fly_df = pd.concat([fly_df,tr_df],axis=1)
            
            
            if get_saccades:
                # make a data structure of saccade times in the same format as the 
                # fly_df trace information
                # data = saccade start times. now not trying to define saccade stops
                # rows = saccade number
                # columns = fly, trial index, trial type
                
                 iterables = [[fly_name],
                             [tr],
                             [this_stim_type]]
                 column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type']) 
                                                             
                 saccade_starts = find_saccades(self.lmr[this_start:this_stop])
                 tr_saccade_starts_df = pd.DataFrame(np.transpose(saccade_starts),columns=column_labels)            
                 fly_saccades_df = pd.concat([fly_saccades_df,tr_saccade_starts_df],axis=1)
            
        return fly_df, fly_saccades_df 
        
        
     
        
    

#---------------------------------------------------------------------------#
def moving_average(values, window):
    #next add gaussian, kernals, etc
    #pads on either end to return an equal length structure,
    #although the edges are distorted
    
    if (window % 2): #is odd 
        window = window + 1; 
    halfwin = window/2
    
    n_values = np.size(values)
    
    padded_values = np.ones(n_values+window)*np.nan
    padded_values[0:halfwin] = np.ones(halfwin)*np.mean(values[0:halfwin])
    padded_values[halfwin:halfwin+n_values] = values
    padded_values[halfwin+n_values:window+n_values+1] = np.ones(halfwin)*np.mean(values[-halfwin:n_values])
  
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(padded_values, weights, 'valid')
    return sma[0:n_values]
    
def xcorr(a, v):
    a = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    v = (v - np.mean(v)) /  np.std(v)
    xc = np.correlate(a, v, mode='same')
    return xc
    
def read_abf(abf_filename):
        fh = AxonIO(filename=abf_filename)
        segments = fh.read_block().segments
    
        if len(segments) > 1:
            print 'More than one segment in file.'
            return 0

        analog_signals_ls = segments[0].analogsignals
        analog_signals_dict = {}
        for analog_signal in analog_signals_ls:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

        return analog_signals_dict
        
def process_wings(raw_wings):
    #here shift wing signal -12 ms in time, filling end with nans
    shifted_wings = np.zeros_like(raw_wings)
    shifted_wings[0:-12] = raw_wings[12:]
    shifted_wings[-11:] = raw_wings[-1]   
    
    #now multiply to convert volts to degrees
    processed_wings = -45 + shifted_wings*33.75
    return processed_wings
     
def find_saccades(raw_lmr_trace,test_plot=False):
    #first fill in nans with nearest signal
    lmr_trace = raw_lmr_trace[~np.isnan(raw_lmr_trace)] 
        #this may give different indexing than input
        #ideally fill in nans in wing processing

    # filter lmr signal
    filtered_trace = butter_lowpass_filter(lmr_trace) #6 hertz
    
    # differentiate, take the absolute value
    diff_trace = abs(np.diff(filtered_trace))
     
    # mark saccade start times -- this could be improved
    diff_thres = .01
    cross_d_thres = np.where(diff_trace > diff_thres)[0]
    
    # #use this to find saccade stops
#     saccade_start_candidate = diff_trace[1:-1] < diff_thres  
#     saccade_cont  = diff_trace[2:]   >= diff_thres
#     stacked_start_cont = np.vstack([saccade_start,saccade_cont])
#     candidate_saccade_starts = np.where(np.all(stacked_start_cont,axis=0))[0]
    
    # impose a refractory period for saccades
    d_cross_d_thres = np.diff(cross_d_thres)
    
    refractory_period = .2 * 10000
    if cross_d_thres.size:
        saccade_starts = [cross_d_thres[0]] #include first
        
        #then take those with gaps between saccade events
        other_is = np.where(d_cross_d_thres > refractory_period)[0]+1
        saccade_starts = np.hstack((saccade_starts,cross_d_thres[other_is]))
    else:
        saccade_starts = []
       
    if test_plot:
        fig = plt.figure()
        plt.plot(lmr_trace,'grey')
        plt.plot(filtered_trace,'black')
        plt.plot(1000*diff_trace,'green')
        
        
        plt.plot(cross_d_thres,np.zeros_like(cross_d_thres),'r.')
        plt.plot(saccade_starts,np.ones_like(saccade_starts),'mo')
    
    # return indicies of start and stop times + saccade magnitude 
    return saccade_starts
       
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=6, fs=10000, order=5): #how does the order change?
    b, a = butter_lowpass(cutoff, fs, order)
    #y = sp.signal.lfilter(b, a, data) #what's the difference here? 
    y = sp.signal.filtfilt(b, a, data)
    return y
      
def write_to_pdf(f_name,figures_list):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(fname)
    for f in figures_list:
        pp.savefig(f)
    pp.close()

def plot_many_flies(path_name, filenames_df):    

    #loop through all genotypes
    genotypes = (pd.unique(filenames_df.values[:,1]))
    print genotypes
    
    for g in genotypes:
        these_genotype_indicies = np.where(filenames_df.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + filenames_df.values[index,0])
            title_txt = filenames_df.values[index,1] + '  ' + filenames_df.values[index,0]
            fly.process_fly()
            fly.plot_wba_stim(title_txt)
        
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_kir_looming.png',dpi=100)
            plt.close('all')
                    
def get_pop_traces_df(path_name, population_f_names):  
    #loop through all genotypes
    #structure row = time points, aligned to looming start
    #columns: genotype, fly, trial index, trial typa, lwa/rwa
    #just collect these for all flies
    
    #genotypes must be sorted to the labels for columns 
    genotypes = (pd.unique(population_f_names.values[:,1]))
    genotypes = np.sort(genotypes)
    genotypes = genotypes[1:]
    print genotypes
    
    population_df = pd.DataFrame()
    
    #loop through each genotype  
    for g in genotypes:
        g
        these_genotype_indicies = np.where(population_f_names.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + population_f_names.values[index,0])
            fly.process_fly()
            fly_df = fly.get_traces_by_stim(g)
            population_df = pd.concat([population_df,fly_df],axis=1)
    return population_df
     
def plot_pop_flight_behavior_histograms(population_df, wba_lim=[-3,3],cnds_to_plot=range(9)):  
    #for the looming data, plot histograms over time of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_cell_names = population_df.loc[:,(g)].columns.get_level_values(0)
        n_cells = np.size(np.unique(all_cell_names))
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
     
            #plot WBA histogram signal -----------------------------------------------------------    
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
        
            #just plot the mean for debugging
            #wba_ax.plot(np.nanmean(g_lmr,1))
        
            #now plot the histograms over time. ------------
            max_t = np.shape(g_lmr)[0]
            n_trs = np.shape(g_lmr)[1]
                     
            t_points = range(max_t)
            t_matrix = np.tile(t_points,(n_trs,1))
            t_matrix_t = np.transpose(t_matrix)

            t_flat = t_matrix_t.flatten() 
            g_lmr_flat = g_lmr.flatten()

            #now remove nans
            g_lmr_flat = g_lmr_flat[~np.isnan(g_lmr_flat)]
            t_flat = t_flat[~np.isnan(g_lmr_flat)]

            #calc, plot histogram
            h2d, xedges, yedges = np.histogram2d(t_flat,g_lmr_flat,bins=[200,50],range=[[0, 4200],[-3,3]],normed=True)
            wba_ax.pcolormesh(xedges, yedges, np.transpose(h2d))
        
           
            #plot white line for 0 -----------
            wba_ax.axhline(color=white)
        
            wba_ax.set_xlim(x_lim) 
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_histograms.png',dpi=100)
        #plt.close('all')

def plot_pop_flight_behavior_means(population_df, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=black,linewidth=.5)        
        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=magenta,linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)
        
            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_means.png',dpi=100)
        plt.close('all')
        
def plot_pop_flight_behavior_means_overlay(population_df, two_genotypes, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    fig = plt.figure(figsize=(16.5, 9))
    #change this so I'm not hardcoding the number of axes
    gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
    genotype_colors = [magenta, blue]
    
    i = 0 
    title_txt = '';
    for g in two_genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = title_txt + g + ' __ ' + str(n_cells) + ' flies ' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
        
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=.25)        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)

            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)

            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
          
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])

            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=black)

            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 

            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        i = i + 1
        
    #now annotate        
    fig.text(.06,.8,'left',fontsize=14)
    fig.text(.06,.53,'center',fontsize=14)
    fig.text(.06,.25,'right',fontsize=14)
    
    fig.text(.22,.905,'22 l/v',fontsize=14)
    fig.text(.495,.905,'44 l/v',fontsize=14)
    fig.text(.775,.905,'88 l/v',fontsize=14)        

    fig.text(.1,.95,two_genotypes[0],color='magenta',fontsize=18)
    fig.text(.2,.95,two_genotypes[1],color='blue',fontsize=18)
    plt.draw()
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path + title_txt + '_population_kir_looming_means_overlay_' 
        + two_genotypes[0] + '_' + two_genotypes[1] + '.png',dpi=100)
    #plt.close('all')


    
