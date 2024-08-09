# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:14:47 2024

@author: jgilbert
"""

""" script to extract demands and deliveries for comparison by 
    water use or contract type in CalLite

"""
import os, sys
import pandas as pnd
idx = pnd.IndexSlice
sys.path.append(r'D:\02_Projects\CalSim\util\CalSim_Utilities\calsim3_object')
import cs3  #<--- a set of Calsim-specific functions, mainly for working with DSS files in a CalSim context
import cs_util as util  #<-- additional utilities related to CalSim
import csPlots     #<-- some plotting and related data transformation functions
import datetime as dt
import numpy as np

#%% The ultimate goal is to calculate the unmet demand and related reliability
#   consistency, and equity metrics. To do that we need to know what the demands
#   are and the corresponding deliveries. In cases where we changed the demands
#   we'll need to read in and use the modified demands.
# The code in this cell lists the delivery and demand variables to be read in


del_vars = ['/DEL_CVP_PAG_S/',
            '/DEL_CVP_PAG_N/',
            '/DEL_CVP_PEX_S/',
            '/DEL_CVP_PSC_N/',
            '/DEL_CVP_PMI_N/',
            '/DEL_CVP_PMI_S/',
            '/DEL_CVP_PRF_S/',
            '/DEL_CVP_PRF_N/',
            '/DEL_SWP_PAG/',
            '/DEL_SWP_MWD/']


alloc_vars = ['/PERDV_CVPAG_S/',
              '/PERDV_CVPAG_SYS/',
              '/PERDV_CVPEX_S/',
              '/PERDV_CVPSC_SYS/',
              '/PERDV_CVPMI_SYS/',
              '/PERDV_CVPMI_S/',
              '/PERDV_CVPRF_SYS/',
              '/PERDV_CVPRF_S/',
              '/PERDV_SWP_AG1/',
              '/PERDV_SWP_MWD1/']

dem_vars = ['/ACVPDEM_PAG_UDEF_SDV/',
            '/ACVPDEM_PAG_UDEF_SYSDV/',
            '/ACVPDEM_PEX_UDEF_SYSDV/',
            '/ACVPDEM_PSC_UDEF_SYSDV/',
            '/ACVPDEM_PMI_UDEF_SYSDV/',
            '/ACVPDEM_PMI_UDEF_SDV/',
            '/ACVPDEM_PRF_UDEF_SYSDV/',
            '/ACVPDEM_PRF_UDEF_SDV/',
            '/SWP_AG_UDEF_DV/',
            '/SWP_MWD_UDEF_DV/']

nod_pag_demvars = ['CON_D171_PAG','CON_D172_PAG','CON_D174_PAG','CON_D178_PAG','DEM_D104_PAG']
sod_pag_demvars = ['ACVPDEM_PAG_UDEF_SDV']
pag_del_vars = ['/DEL_CVP_PAG_S/', '/DEL_CVP_PAG_N/']


#%% Set the location of the ensemble datasets
#  *_run_listing.csv files should contain a listing of the CalLite scenarios to
#  be read in

ens_base_dir = r'D:\02_Projects\CalSim\proj\2023_CalClimateInit\calsim\CalLite_Exploratory03'
ofp = os.path.join(ens_base_dir,'analysis','callite_expl02b_run_listing.csv')
ofp = os.path.join(ens_base_dir,'analysis','callite_expl03_run_listing.csv')
run_df = pnd.read_csv(ofp,header=[0], index_col=0)

outdir = os.path.join(ens_base_dir, 'analysis')
todaystr = dt.date.today().strftime('%Y%m%d')
version_name = f'{todaystr}draft'
#%% Iterate through all runs - read in data

for study, indic in run_df.iterrows():

    study_sht= 'CalLite'+study.split('_')[-1].upper()
    #base_calsim_launch = fr'D:/02_Projects/CalSim/proj/2023_CalClimateInit/calsim/Exploratory_v01/{study}/{study}.launch'
    base_calsim_launch = fr'D:/02_Projects/CalSim/proj/2023_CalClimateInit/calsim/CalLite_Exploratory03/{study}/{study}.launch'
    
    c1 = cs3.calsim(launchFP = base_calsim_launch)
    
#% all dem vars
    c1.DVdata.getDVts(filter=dem_vars)
    udef_demDV = c1.DVdata.DVtsDF.copy(deep=True)
    c1.SVdata.getSVts(filter=['/ACVPDEM_PAG_S/','/ACVPDEM_PAG_SYS/',  # read in the original demands data in SV file, for reference
                              '/ACVPDEM_PEX_S/','/ACVPDEM_PEX_SYS/',
                              '/ACVPDEM_PLS_S/','/ACVPDEM_PLS_SYS/',
                              '/ACVPDEM_PMI_S/','/ACVPDEM_PMI_SYS/',
                              '/ACVPDEM_PRF_S/','/ACVPDEM_PRF_SYS/',
                              '/ACVPDEM_PSC_SYS/','/ACVPDEM_TOTAL_S/',
                              '/ACVPDEM_TOT_SYS/'])
    demSV =  c1.SVdata.SVtsDF.copy(deep=True)
    
    # calculate the ratio of modified demands to original (this is a check on the data)
    pag_ratio_n = udef_demDV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PAG_UDEF_SYSDV']].values / \
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PAG_SYS']].values-
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PAG_S']].values - 
             udef_demDV.loc['1921-10-31':'2003-09-30', idx[:, 'ACVPDEM_PAG_UDEF_SDV']].values))
    pag_ratio_n = round(np.nanmax(pag_ratio_n),4)
    
    # calculate the ratio of modified demands to original (this is a check on the data)
    pmi_ratio_n = udef_demDV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PMI_UDEF_SYSDV']].values / \
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PMI_SYS']].values-
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PMI_S']].values - 
             udef_demDV.loc['1921-10-31':'2003-09-30', idx[:, 'ACVPDEM_PMI_UDEF_SDV']].values))
    pmi_ratio_n = round(np.nanmax(pmi_ratio_n),4)
    
    # calculate the ratio of modified demands to original (this is a check on the data)
    prf_ratio_n = udef_demDV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PRF_UDEF_SYSDV']].values / \
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PRF_SYS']].values-
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PRF_S']].values - 
             udef_demDV.loc['1921-10-31':'2003-09-30', idx[:, 'ACVPDEM_PRF_UDEF_SDV']].values))
    prf_ratio_n = round(np.nanmax(prf_ratio_n),4)
    
    # calculate the ratio of modified demands to original (this is a check on the data)
    psc_ratio_n = udef_demDV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PSC_UDEF_SYSDV']].values / \
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PSC_SYS']].values)
    psc_ratio_n = round(np.nanmax(psc_ratio_n),4)
    
    # calculate the ratio of modified demands to original (this is a check on the data)
    pex_ratio_s = udef_demDV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PEX_UDEF_SYSDV']].values / \
            (demSV.loc['1921-10-31':'2003-09-30', idx[:,'ACVPDEM_PEX_S']].values)
    pex_ratio_s = round(np.nanmax(pex_ratio_s),4)
    
    
    # Now read in NOD, SOD cvp ag demands and deliveries from DSS file
    noddem = ['/'+v+'/' for v in nod_pag_demvars]
    soddem = ['/'+v+'/' for v in sod_pag_demvars]
    
    c1.DVdata.getDVts(filter=soddem+pag_del_vars)
    c1.SVdata.getSVts(filter=noddem)
    cl_dat = c1.DVdata.DVtsDF.copy(deep=True)
    clsv_dat = c1.SVdata.SVtsDF.copy(deep=True)
    
    # convert from CFS to TAF, then aggregate from monthly to annual volumes on 
    # contract year (March - February) of each year
    nod_ag_dem_df = pnd.DataFrame()
    for c in clsv_dat.columns:
        if c[-1]=='CFS':
            newcol = list(c)
            newcol[-1] = 'TAF'
            newcol = tuple(newcol)
            tmp = csPlots.cfs_to_taf(clsv_dat.loc[:,[c]]) #<-- function to convert to TAF from CFS
            nod_ag_dem_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
        else:
            nod_ag_dem_df[c] = clsv_dat.loc[:, c].resample('A-Feb').apply('sum')
    nod_ag_dem_df.columns = pnd.MultiIndex.from_tuples(nod_ag_dem_df.columns, names = tmp.columns.names)
    # modify the header information to match the new variable information, units
    nod_ag_dem_df[('CALCULATED','Total_NOD_PAG_Demand_TAF','DEMAND','1MON','2020D09E','PER-AVER','TAF') ] = nod_ag_dem_df.sum(axis=1)*pag_ratio_n
    nod_ag_dem_df = nod_ag_dem_df.loc['1922-10-31':'2003-09-30']
    
    sod_ag_dem_df = cl_dat.loc[[i for i in cl_dat.index if i.month==2],idx[:,'ACVPDEM_PAG_UDEF_SDV']]
    
    ag_delivs_df = pnd.DataFrame() 
    for c in cl_dat.columns:
        if c[1][0:11]=='DEL_CVP_PAG':
            if c[-1]=='CFS':
                newcol = list(c)
                newcol[-1] = 'TAF'
                newcol = tuple(newcol)
                tmp = csPlots.cfs_to_taf(cl_dat.loc[:,[c]])
                ag_delivs_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
            else:
                ag_delivs_df[c] = cl_dat.loc[:, c].resample('A-Feb').apply('sum')
    ag_delivs_df.columns = pnd.MultiIndex.from_tuples(ag_delivs_df.columns, names = tmp.columns.names)
    
    all_dat_df = sod_ag_dem_df 
    all_dat_df = all_dat_df.join(nod_ag_dem_df.loc[:, nod_ag_dem_df.columns[-1]]) #, on=nod_ag_dem_df.index) #['Total_NOD_PAG_Demand_TAF'] = nod_ag_dem_df.loc[:,['Total_NOD_PAG_Demand_TAF']]
    all_dat_df = all_dat_df.join(ag_delivs_df)


   #% do the same for Settlement and Exchange contractors (PSC & PEX)
    psc_pex_dem_del = ['/ACVPDEM_PEX_UDEF_SYSDV/','/DEM_CVP_PSC_N_UDEF/', '/ACVPDEM_PSC_UDEF_SYSDV/',
                       '/DEL_CVP_PEX_S/',
                       '/DEL_CVP_PSC_N/']
    
    c1.DVdata.getDVts(filter=psc_pex_dem_del)
    cl_dat2 = c1.DVdata.DVtsDF.copy(deep=True)
    
    
    psc_pex_dem_del_df = pnd.DataFrame() 
    for c in cl_dat2.columns:
        if c[1][0:8]=='ACVPDEM_':
            # pick single value rather than accumulate - annual PSC and PEX demands are repeated monthly 
            # you'll get erroneous numbers if you sum this variable over a year
            psc_pex_dem_del_df[c] = cl_dat2.loc[[i for i in cl_dat2.index if i.month==2], c]
        else:
            if c[-1]=='CFS':
                newcol = list(c)
                newcol[-1] = 'TAF'
                newcol = tuple(newcol)
                tmp = csPlots.cfs_to_taf(cl_dat2.loc[:,[c]])
                psc_pex_dem_del_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
            else:
                psc_pex_dem_del_df[c] = cl_dat2.loc[:, c].resample('A-Feb').apply('sum')
    psc_pex_dem_del_df.columns = pnd.MultiIndex.from_tuples(psc_pex_dem_del_df.columns,
                                                            names = tmp.columns.names)

    # M&I demands need to be summed manually from component variables
    # this section is for CVP NOD M&I
    cvp_pmi_n_dem_vars = ['dem_d167B_PMI_A',
                      'dem_D104_PMI',
                      'dem_d8b_PMI_ann',
                      'dem_d8e_pmi_ann',
                      'dem_d8f_pmi_ann',
                      'dem_d8g_pmi_ann',
                      'dem_d8h_pmi_ann',
                      'dem_d8i_pmi_ann',
                      'dem_d9ab_pmi_ann',
                      'dem_d9b_pmi_ann']
    cvp_pmi_n_del_vars = ['D167B_PMI','D104_PMI','D8_PMI','D9_PMI']
    
    c1.SVdata.getSVts(filter=[f'/{c.upper()}/' for c in cvp_pmi_n_dem_vars])
    pmi_dem_n = c1.SVdata.SVtsDF.copy(deep=True)
    
    c1.DVdata.getDVts(filter=[f'/{c.upper()}/' for c in cvp_pmi_n_del_vars])
    pmi_del_n = c1.DVdata.DVtsDF.copy(deep=True)
    
    dels_ann = pnd.DataFrame()
    
    for c in pmi_del_n.columns:
        newcol = list(c)
        newcol[-1] = 'TAF'
        newcol = tuple(newcol)
        tmp = csPlots.cfs_to_taf(pmi_del_n.loc[:,[c]])
        dels_ann[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
    dels_ann.columns = pnd.MultiIndex.from_tuples(dels_ann.columns, names = tmp.columns.names)
    
    pmi_D104_dem_n_ann = pmi_dem_n.loc[:,idx[:,'DEM_D104_PMI']].resample('A-Feb').apply('sum') 
    pmi_dem_n_ann = pmi_dem_n.loc[[i for i in cl_dat.index if i.month==2], 
                                  [c for c in pmi_dem_n.columns if c[1]!='DEM_D104_PMI']]
    pmi_dem_n_ann[pmi_D104_dem_n_ann.columns[0]] = pmi_D104_dem_n_ann
    pmi_dem_n_ann = pmi_dem_n_ann.sum(axis=1)*pmi_ratio_n #<-- in case M&I demands were adjusted, they can be accounted for here
    
    
    c1.DVdata.getDVts(filter=['/ACVPDEM_PMI_UDEF_SYSDV/','/ACVPDEM_PMI_UDEF_SDV/',
                              '/DEL_CVP_PMI_N/','/DEL_CVP_PMI_S/'])
    cvp_pmi_= c1.DVdata.DVtsDF.copy(deep=True)
    
    cvp_pmi_df = pnd.DataFrame() 
    for c in cvp_pmi_.columns:
        if c[1][0:8]=='ACVPDEM_':
            # pick single value rather than accumulate
            cvp_pmi_df[c] = cvp_pmi_.loc[[i for i in cvp_pmi_.index if i.month==2], c]
        else:
            if c[-1]=='CFS':
                newcol = list(c)
                newcol[-1] = 'TAF'
                newcol = tuple(newcol)
                tmp = csPlots.cfs_to_taf(cvp_pmi_.loc[:,[c]])
                cvp_pmi_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
            else:
                cvp_pmi_df[c] = cvp_pmi_.loc[:, c].resample('A-Feb').apply('sum')
    cvp_pmi_df.columns = pnd.MultiIndex.from_tuples(cvp_pmi_df.columns,
                                                            names = tmp.columns.names)
    
    cvp_pmi_df[('CALCULATED','Total_NOD_PMI_Demand_TAF','DEMAND','1MON','2020D09E','PER-AVER','TAF')] = pmi_dem_n_ann #pnd.MultiIndex.from_tuples(('CA'))]
    cvp_pmi_df[('CALCULATED','Total_SOD_PMI_Delivery_TAF','DELIVERY','1MON','2020D09E','PER-AVER', 'TAF')] = dels_ann.sum(axis=1)

   #% Calculate deliveries and demands for CVP refuges, NOD and SOD
    prf_del_dems = ['/ACVPDEM_PRF_UDEF_SDV/','/ACVPDEM_PRF_UDEF_SYSDV/', '/DEL_CVP_PRF_N/','/DEL_CVP_PRF_S/']
    c1.DVdata.getDVts(filter=prf_del_dems)
    cvp_prf_= c1.DVdata.DVtsDF.copy(deep=True)
    
    cvp_prf_df = pnd.DataFrame() #cl_dat.loc[:,[ic for ic in cl_dat.columns if ic[1][0:11]=='DEL_CVP_PAG']]
    for c in cvp_prf_.columns:
        if c[1][0:8]=='ACVPDEM_':
            # pick single value rather than accumulate
            cvp_prf_df[c] = cvp_prf_.loc[[i for i in cvp_prf_.index if i.month==2], c]
        else:
            if c[-1]=='CFS':
                newcol = list(c)
                newcol[-1] = 'TAF'
                newcol = tuple(newcol)
                tmp = csPlots.cfs_to_taf(cvp_prf_.loc[:,[c]])
                cvp_prf_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
            else:
                cvp_prf_df[c] = cvp_prf_.loc[:, c].resample('A-Feb').apply('sum')
    cvp_prf_df.columns = pnd.MultiIndex.from_tuples(cvp_prf_df.columns,
                                                            names = tmp.columns.names)
    cvp_prf_df[('CALCULATED','Total_NOD_PRF_Demand_TAF','DEMAND','1MON','2020D09E','PER-AVER', 'TAF')] = \
        cvp_prf_df.loc[:,idx[:,'ACVPDEM_PRF_UDEF_SYSDV']].values - cvp_prf_df.loc[:,idx[:,'ACVPDEM_PRF_UDEF_SDV']].values


    #% Now calculate SWP Ag, M&I to MWD, and M&I to all others (SWP_OTH) 
    swp_vars = ['/SWP_AG_UDEF_DV/','/SWP_MWD_UDEF_DV/','/SWP_OTH_UDEF_DV/',
                '/DEL_SWP_PAG_S/','/DEL_SWP_MWD/','/DEL_SWP_OTH/']
    
    c1.DVdata.getDVts(filter=swp_vars)
    swp_= c1.DVdata.DVtsDF.copy(deep=True)
    
    swp_df = pnd.DataFrame() 
    for c in swp_.columns:
        if 'UDEF' in c[1]:
            # pick single value rather than accumulate
            swp_df[c] = swp_.loc[[i for i in swp_.index if i.month==2], c]
        else:
            if c[-1]=='CFS':
                newcol = list(c)
                newcol[-1] = 'TAF'
                newcol = tuple(newcol)
                tmp = csPlots.cfs_to_taf(swp_.loc[:,[c]])
                swp_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
            else:
                swp_df[c] = swp_.loc[:, c].resample('A-Feb').apply('sum')
    swp_df.columns = pnd.MultiIndex.from_tuples(swp_df.columns, names = tmp.columns.names)


    #% Delta demands & deliveries are contained in another set of variables
    # list them here
    delta_vars = ['/DEMAND_D_BrananIs_NP/','/DEMAND_D_SacSJR_NP/',
                  '/DEMAND_D_Terminous_NP/','/DEMAND_D_Stockton_NP/',
                  '/DEMAND_D_MedfordIs_NP/','/DEMAND_D_ConeyIs_NP/',
                  '/D_BrananIs/','/D_SACSJR/', '/D_Terminous/','/D_Stockton/',
                  '/D_MedfordIs/','/D_ConeyIs/']
    
    # now retrieve those data
    c1.DVdata.getDVts(filter=delta_vars)
    delta_ = c1.DVdata.DVtsDF.copy(deep=True)
    
    # convert to TAF and aggregate by contract year
    delta_df = pnd.DataFrame()
    for c in delta_.columns:
        if c[-1]=='CFS':
            newcol = list(c)
            newcol[-1] = 'TAF'
            newcol = tuple(newcol)
            tmp = csPlots.cfs_to_taf(delta_.loc[:,[c]])
            delta_df[tmp.columns[0]] = tmp.loc[:, tmp.columns[0]].resample('A-Feb').apply('sum')
        else:
            delta_df[c] = delta_.loc[:, c].resample('A-Feb').apply('sum')
    
    dem_cols = [c for c in delta_df.columns if 'DEMAND' in c[1].upper()]
    del_cols = [c for c in delta_df.columns if 'DELIVERY' in c[2].upper()]
    
    delta_df[('CALCULATED','Total_Delta_Demand_TAF','DEMAND','1MON','2020D09E','PER-AVER', 'TAF')] = \
         delta_df.loc[:, dem_cols].sum(axis=1)
         
    delta_df[('CALCULATED','Total_Delta_Delivery_TAF','DELIVERY','1MON','2020D09E','PER-AVER', 'TAF')] = \
         delta_df.loc[:, del_cols].sum(axis=1)
    
    #% combine all the data together into one dataframe
    all_dat_df = all_dat_df.join(psc_pex_dem_del_df) #, on=psc_pex_dem_del_df.index)
    all_dat_df = all_dat_df.join(cvp_pmi_df.iloc[:, [0,2,3,4]])
    all_dat_df = all_dat_df.join(cvp_prf_df.iloc[:, [0,2,3,4]])
    all_dat_df = all_dat_df.join(swp_df)
    all_dat_df = all_dat_df.join(delta_df)
    
    # write this dataframe out to CSV for use in the next step
    ofp = os.path.join(outdir, version_name +f'_{study_sht}_DelivsDemands.csv')
    all_dat_df.to_csv(ofp, header=True)
    


