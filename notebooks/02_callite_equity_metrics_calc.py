# -*- coding: utf-8 -*-
"""
This script contains a number of different metrics calculations using
CalLite scenario outputs - much of this may be extraneous according to
current workflows 

Created on Sun Feb  4 21:34:09 2024

@author: jgilbert
"""
import os, sys
import pandas as pnd
idx = pnd.IndexSlice
import numpy as np
import datetime as dt

sys.path.insert(0,r'D:\02_Projects\CalSim\util\CalSim_Utilities\calsim3_object')

import cs3
import cs_util as util
import AuxFunctions as af
import csPlots

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr

def dict_to_df(dat):
    dat_ = []
    for ri, rv in dat.items():
        cols = rv.columns[0]
        new_df = rv #pnd.DataFrame()
        new_df[(ri,cols[1],cols[2],cols[3],cols[4],cols[5],cols[6])] = rv.values
        dat_.append(new_df.iloc[:, 1])
        #dat_df[(ri,cols[1],cols[2],cols[3],cols[4],cols[5],cols[6]) ] = rv.values  
        
    dat_df = pnd.concat(dat_, axis=1)
    dat_df.columns = dat_df.columns.set_names(rv.columns.names, level=None)
    return(dat_df)

def res_sto_metric(df, month, level, run_df, label):
    
    sel_ix = [i for i in df.index if i.month in month]
    #for c in df.columns:
    thisdat = df.loc[sel_ix, :]
    
    df_lev = np.percentile(thisdat, level, axis=0)
    df_fin= run_df.copy(deep=True)
    df_fin[label] = df_lev
    
    return(df_fin)

def mon_to_ann(df,ann_on='A-Sep'):

    df_ = pnd.DataFrame()
    tmp_df = []
    for c in df.columns:
        if c[-1]=='CFS':
            newcol = list(c)
            newcol[-1] = 'TAF'
            newcol = pnd.MultiIndex.from_tuples((tuple(newcol,),))
            tmp = csPlots.cfs_to_taf(df.loc[:,[c]])
            #df_.loc[:, newcol] = tmp.loc[:, [tmp.columns[0]]].resample(ann_on).apply('sum')
            tmp_df.append(tmp.loc[:, [tmp.columns[0]]].resample(ann_on).apply('sum'))
        else:
            #df_[c] = df.loc[:, c].resample(ann_on).apply('sum') 
            tmp_df.append(df.loc[:, c].resample(ann_on).apply('sum') )
    df_ = pnd.concat(tmp_df, axis=1)
    
    return(df_)

def rescale(d, minv=0, maxv=1):
    
    dmax = np.max(d)
    dmin = np.min(d)
    drescale = [((di-dmin)/(dmax-dmin))*maxv for di in d.values]
    drescale = pnd.Series(drescale, index=d.index)
    drescale.name = d.name
    return(drescale)
    
#%% Set the directory and file location - modify the `todaystr` variable
#    to correspond to the prefix used in 01_get_delivery_demands_callite.py and
#    the csv file written by that script
ens_base_dir = r'D:\02_Projects\CalSim\proj\2023_CalClimateInit\calsim\CalLite_Exploratory02'
ens_base_dir = r'D:\02_Projects\CalSim\proj\2023_CalClimateInit\calsim\CalLite_Exploratory03'
ofp = os.path.join(ens_base_dir,'analysis','callite_expl03_run_listing.csv')
run_df = pnd.read_csv(ofp,header=[0], index_col=0)

todaystr = f"{dt.date.today().strftime('%Y%m%d')}DRAFT"
todaystr = '20240605DRAFT' #<--- modify this if needed to match a previously extracted CSV, otherwise comment out

#%% get the total 8-river unimpaired flow as a measure of wet/dry
i='expl0000' # just pick the baseline for this
lfp = os.path.join(ens_base_dir,f'{i}',f'{i}.launch' )

c1 = cs3.calsim(launchFP = lfp)
c1.SVdata.getSVts(filter=['UNIMP_'])
unimpDF = c1.SVdata.SVtsDF
saccols = [c for c in unimpDF.columns if c[1] in ['UNIMP_SRBB','UNIMP_YUBA','UNIMP_FOLS','UNIMP_OROV']]
sjrcols = [c for c in unimpDF.columns if c[1] in ['UNIMP_SJ','UNIMP_ME','UNIMP_TU','UNIMP_ST']]
unimpDF[('CALCULATED','SAC_UNIMP','FLOW-UNIMPAIRED','1MON','2020D09E','PER-AVER','TAF')] = unimpDF.loc[:, saccols].sum(axis=1)
unimpDF[('CALCULATED','SJR_UNIMP','FLOW-UNIMPAIRED','1MON','2020D09E','PER-AVER','TAF')] = unimpDF.loc[:, sjrcols].sum(axis=1)
unimpDF[('CALCULATED','EIGHTRIV_UNIMP','FLOW-UNIMPAIRED','1MON','2020D09E','PER-AVER','TAF')] = unimpDF.iloc[:, -2:].sum(axis=1)

unimp_ann = unimpDF.resample('A-Sep').apply(np.sum)

ofp2 = os.path.join(ens_base_dir,'analysis','callite_expl_unimpairedflow_8riv.csv')

unimp_ann.to_csv(ofp2, header=True)

c1.DVdata.getDVts(filter=['/WYT_SAC_/'])
sac_wyt = c1.DVdata.DVtsDF
dry_yr_idx = sac_wyt[sac_wyt.columns[0]]>=4
wet_yr_idx = sac_wyt[sac_wyt.columns[0]]<4
#%% let's just try getting shasta storage from all the runs

sha = {}
sacin = {}
sjrin = {}
deltain = {}
ndo = {}

orovl = {}
folsm = {}
melon = {}
mlrtn = {}
trnty = {}

banks = {}
jones = {}

x2 = {}
jp_ec = {}
rs_ec = {}

del_cvpn = {}
del_cvps = {}
del_cvpagn = {}
del_cvpags = {}
del_cvpsc = {}
del_cvpex = {}
del_cvpmin = {}
del_cvpmis = {}
del_cvprfn = {}
del_cvprfs = {}
del_swp_mwd ={}
del_swp_oth ={}
del_swp_ag ={}

varlist = ['/S_SHSTA/','/S_OROVL/','/S_TRNTY/','/S_FOLSM/','/S_MELON/',
           '/S_MLRTN/','/C_HOOD/', '/I_HOOD_S2D/','/C_SJRVER/','/AD_SJRCalAll/',
           '/INFLOW/INFLOW-DELTA/','/NDO/','/DEL_CVP_TOTAL_N/','/DEL_CVP_TOTAL_S/',
           '/DEL_CVP_PAG_N/','/DEL_CVP_PAG_S/','/DEL_CVP_PSC_N/','/DEL_CVP_PEX_S/',
           '/DEL_CVP_PRF_N/','/DEL_CVP_PRF_S/','/DEL_SWP_MWD/','/DEL_SWP_OTH/',
           '/DEL_SWP_PAG/','/JP_EC_MONTH/','/RS_EC_MONTH/','/X2_PRV/']
           

   
    
#%% delivery reliability/consistency - following metrics proposed by Siddiqi, A., 
# Wescoat, J. L., & Muhammad, A. (2018). Socio-hydrological assessment of water 
# security in canal irrigation systems: A conjoint quantitative analysis of equity 
#and reliability. Water Security, 4–5, 44–55. https://doi.org/10.1016/j.wasec.2018.11.001

deldempref = '20240605draft_CalLite'

deldem_pairs = {'ACVPDEM_PAG_UDEF_SDV':'DEL_CVP_PAG_S',
                'Total_NOD_PAG_Demand_TAF': 'DEL_CVP_PAG_N',
                'ACVPDEM_PEX_UDEF_SYSDV': 'DEL_CVP_PEX_S',
                'ACVPDEM_PSC_UDEF_SYSDV': 'DEL_CVP_PSC_N',  #DEM_CVP_PSC_N_UDEF
                'ACVPDEM_PMI_UDEF_SDV': 'DEL_CVP_PMI_S',
                'Total_NOD_PMI_Demand_TAF': 'DEL_CVP_PMI_N',
                'ACVPDEM_PRF_UDEF_SDV':'DEL_CVP_PRF_S',
                #'Total_NOD_PRF_Demand_TAF':'DEL_CVP_PRF_N',
                'SWP_MWD_UDEF_DV':'DEL_SWP_MWD',
                'SWP_OTH_UDEF_DV':'DEL_SWP_OTH',
                'SWP_AG_UDEF_DV':'DEL_SWP_PAG_S',
                'CVP_All_Ag_Demand': 'CVP_All_Ag_Deliv'}
#%%
synth_metrics = {} 
for r, p in run_df.iterrows():
    # read in the processed demand/deliveyr data from `01_get_delivery_demands_callite.py`
    fp = os.path.join(ens_base_dir, 'analysis', deldempref+r.upper()+'_DelivsDemands.csv')

    deldem = pnd.read_csv(fp, header=[0,1,2,3,4,5,6], index_col=0, 
                          parse_dates=True, infer_datetime_format=True)    
    
    deldem_ratio_df = pnd.DataFrame(index=deldem.index)
    resilfailreliab = {}
    
    succ_thresh = 0.5
    for k,v in deldem_pairs.items():
        if k =='CVP_All_Ag_Demand':
            totdem = deldem.loc[:, idx[:, ('ACVPDEM_PAG_UDEF_SDV','Total_NOD_PAG_Demand_TAF',
                                     'ACVPDEM_PSC_UDEF_SYSDV','ACVPDEM_PEX_UDEF_SYSDV')]].sum(axis=1)
            totdel = deldem.loc[:, idx[:, ('DEL_CVP_PAG_S','DEL_CVP_PAG_N',
                                     'DEL_CVP_PEX_S','DEL_CVP_PSC_N')]].sum(axis=1)
            deldem_ratio = totdel/totdem
            deldem_ratio_df[v] = [min(1.0, dd) for dd in deldem_ratio]
            

            succ_fail = deldem_ratio[1:]>succ_thresh      
            w_ = []
            for ii, dd in enumerate(deldem_ratio[1:-1]):
                if dd>=succ_thresh and deldem_ratio[ii+2]<succ_thresh: 
                    # a success followed by a failure
                    w_.append(1)
                else:
                    w_.append(0)
            alpha_ = np.mean(succ_fail)
            rho_ = np.mean(w_)
        else:
            deldem_ratio = deldem.loc[:, idx[:,v]].values/(deldem.loc[:, idx[:,k]].values)
            deldem_ratio_df[v] = [min(1.0,dd[0]) for dd in deldem_ratio]
        
            succ_fail = deldem_ratio[1:]>succ_thresh      
            w_ = []
            for ii, dd in enumerate(deldem_ratio[1:-1]):
                if dd[0]>=succ_thresh and deldem_ratio[ii+2][0]<succ_thresh: 
                    # a success followed by a failure
                    w_.append(1)
                else:
                    w_.append(0)
            alpha_ = np.mean(succ_fail)
            rho_ = np.mean(w_)
        
        if alpha_==1: # no failures - perfectly resilient
            resiliency_rate = 1.    
        else:
            resiliency_rate = min(1, rho_/(1-alpha_))
            
        if rho_==0:
            fail_length = 0 # no failures
        else:
            fail_length = (1-alpha_)/rho_
        
        resilfailreliab[v] = [resiliency_rate, fail_length, alpha_]
                
      
    selcols = [c for c in deldem_ratio_df.columns if c not in ['CVP_All_Ag_Deliv']]
        
    deldem_ratio_df = deldem_ratio_df.iloc[1:, :] # drop 1922 because of partial contract year
    unmetdemand_CVPagTot = deldem_ratio_df.loc[:, ['CVP_All_Ag_Deliv']].mean(axis=0)
    deldem_ratio_df = deldem_ratio_df.loc[:, selcols]
    
    consistency = np.std(deldem_ratio_df, axis=0)/np.mean(deldem_ratio_df, axis=0)
    equity = np.std(deldem_ratio_df, axis=1)/np.mean(deldem_ratio_df, axis=1)
    
    mean_consistency = np.mean(consistency)
    mean_equity = 1- np.mean(equity) # as calc'd, most equitable value would be 0 - subtract from 1 to make a higher number "better"
    
    resil_cols = [f'Resil_{c}' for c in resilfailreliab.keys()]
    reliab_cols = [f'Reliab_{c}' for c in resilfailreliab.keys()]    
    
    resil_vals = [r[0] for r in resilfailreliab.values()]
    reliab_vals = [r[2] for r in resilfailreliab.values()]
    
    synth_metrics[r] = [mean_equity, mean_consistency, unmetdemand_CVPagTot.iloc[0]] + resil_vals +reliab_vals #['equity'].append(mean_equity)
    #synth_metrics['consistency'].append(mean_consistency)
    
synth_metrics_df = pnd.DataFrame.from_dict(synth_metrics, orient='index', 
                                           columns=['Equity','Consistency', 'Ann_Avg_CVPAgUnmet']+resil_cols+reliab_cols)


#%% Write out results to CSV
ofp = os.path.join(ens_base_dir, 'analysis','summary_data', f'{todaystr}_summary_metrics.csv')
#param_synth_all_df.to_csv(ofp, header=True)
synth_metrics_df.to_csv(ofp, header=True)

#%% separate by wet and dry year types

synth_metrics_sub = {} #'equity':[], 'consistency':[]}

for subset in ['wet','dry']:
    if subset=='wet':
        thisidx = wet_yr_idx
    else:
        thisidx = dry_yr_idx
    for r, p in run_df.iterrows():
    
        # read in the processed demand/deliveyr data from `get_delivery_demands_callite_mekoplots_loop.py`
        fp = os.path.join(ens_base_dir, 'analysis', deldempref+r.upper()+'_DelivsDemands.csv')
    
        deldem = pnd.read_csv(fp, header=[0,1,2,3,4,5,6], index_col=0, 
                              parse_dates=True, infer_datetime_format=True)    
        
        deldem = deldem.loc[thisidx,:]
    
        
        deldem_ratio_df = pnd.DataFrame(index=deldem.index)
    
        resilfailreliab = {}
    
        
        succ_thresh = 0.5
        for k,v in deldem_pairs.items():
            if k =='CVP_All_Ag_Demand':
                totdem = deldem.loc[:, idx[:, ('ACVPDEM_PAG_UDEF_SDV','Total_NOD_PAG_Demand_TAF',
                                         'ACVPDEM_PSC_UDEF_SYSDV','ACVPDEM_PEX_UDEF_SYSDV')]].sum(axis=1)
                totdel = deldem.loc[:, idx[:, ('DEL_CVP_PAG_S','DEL_CVP_PAG_N',
                                         'DEL_CVP_PEX_S','DEL_CVP_PSC_N')]].sum(axis=1)
                deldem_ratio = totdel/totdem
                deldem_ratio_df[v] = [min(1.0, dd) for dd in deldem_ratio]
                
    
                succ_fail = deldem_ratio[1:]>succ_thresh      
                w_ = []
                for ii, dd in enumerate(deldem_ratio[1:-1]):
                    if dd>=succ_thresh and deldem_ratio[ii+2]<succ_thresh: 
                        # a success followed by a failure
                        w_.append(1)
                    else:
                        w_.append(0)
                alpha_ = np.mean(succ_fail)
                rho_ = np.mean(w_)
            else:
                deldem_ratio = deldem.loc[:, idx[:,v]].values/(deldem.loc[:, idx[:,k]].values)
                deldem_ratio_df[v] = [min(1.0,dd[0]) for dd in deldem_ratio]
            
                succ_fail = deldem_ratio[1:]>succ_thresh      
                w_ = []
                for ii, dd in enumerate(deldem_ratio[1:-1]):
                    if dd[0]>=succ_thresh and deldem_ratio[ii+2][0]<succ_thresh: 
                        # a success followed by a failure
                        w_.append(1)
                    else:
                        w_.append(0)
                alpha_ = np.mean(succ_fail)
                rho_ = np.mean(w_)
            
            if alpha_==1: # no failures - perfectly resilient
                resiliency_rate = 1.    
            else:
                resiliency_rate = min(1, rho_/(1-alpha_))
                
            if rho_==0:
                fail_length = 0 # no failures
            else:
                fail_length = (1-alpha_)/rho_
            
            resilfailreliab[v] = [resiliency_rate, fail_length, alpha_]
                    
          
        selcols = [c for c in deldem_ratio_df.columns if c not in ['CVP_All_Ag_Deliv']]
            
        deldem_ratio_df = deldem_ratio_df.iloc[1:, :] # drop 1922 because of partial contract year
        unmetdemand_CVPagTot = deldem_ratio_df.loc[:, ['CVP_All_Ag_Deliv']].mean(axis=0)
        deldem_ratio_df = deldem_ratio_df.loc[:, selcols]
            
        consistency = np.std(deldem_ratio_df, axis=0)/np.mean(deldem_ratio_df, axis=0)
        equity = np.std(deldem_ratio_df, axis=1)/np.mean(deldem_ratio_df, axis=1)
        
        mean_consistency = np.mean(consistency)
        mean_equity = 1- np.mean(equity) # as calc'd, most equitable value would be 0 - subtract from 1 to make a higher number "better"
        
        resil_cols = [f'Resil_{c}' for c in resilfailreliab.keys()]
        reliab_cols = [f'Reliab_{c}' for c in resilfailreliab.keys()]    
        
        resil_vals = [r[0] for r in resilfailreliab.values()]
        reliab_vals = [r[2] for r in resilfailreliab.values()]
        
        synth_metrics_sub[r] = [mean_equity, mean_consistency, unmetdemand_CVPagTot.iloc[0]] + resil_vals +reliab_vals #['equity'].append(mean_equity)
        #synth_metrics['consistency'].append(mean_consistency)
    
    synth_metrics_df = pnd.DataFrame.from_dict(synth_metrics_sub, orient='index', 
                                               columns=['Equity','Consistency', 'Ann_Avg_CVPAgUnmet']+resil_cols+reliab_cols)
    
    # param_synth_all_df = pnd.concat([all_dat_df, synth_metrics_df], axis=1)
    

    ofp = os.path.join(ens_base_dir, 'analysis','summary_data', f'{todaystr}_summary_metrics_{subset}.csv')
    #param_synth_all_df.to_csv(ofp, header=True)
    synth_metrics_df.to_csv(ofp, header=True)