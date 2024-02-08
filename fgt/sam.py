import pandas as pd
import geopandas as gpd
import numpy as np
import os

os.chdir(r'c:\users\brenn\documents\Research\sam')

## Create Columns

demographics = ["F0_14", "F65_", "SingleParentFamilies", "LICO_at", "Unemployed", "F30__IncomeToHousing", "SubsidizedHousing", "UnsuitHouse", "RepairHouse", "Indig", "VisMin", "Black", "Imm1980_21", "Imm2016_21", "Refugees", "NoEngFr", "F60_MinCommute"]

# Base Lists
destinations = ['rf', 'ef', 'emp', 'ps', 'hf', 'caf', 'g1', 'g3', 'g5']
modes = ['cyc', 'wlk', 'ptp', 'ptop']
# 'cdvul', 'cmavul', 'csdvul'
geographies = ['', 'cd', 'cma', 'csd']

# modify lists for specific column
demographics = [s + '_num' for s in demographics]

## Define Calculations

def column_format(destination, mode, geography):
    return f'{destination}_ai_{mode}_{geography}'

def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    # calculates population weighted percentiles
    if len(values) == 1:
        return values[0]
    else:
        i = np.argsort(values)
        c = np.cumsum(weights[i])
        q = np.searchsorted(c, quantiles * c[-1])

        if c[q]/c[-1] == 1:
            # solve edge case where entire population is located in one sub-geography
            return values[i[q]]
        else:
            return np.where(c[q]/c[-1] == quantiles, 0.5 * (values[i[q]] + values[i[q+1]]), values[i[q]])

def pov_line(df, geography, mode, population, quantile):
    # poverty line calculation
    id_list = []
    pov_list = []

    # for each geography in table, calculate the population weighted 25%
    for name, group in df.groupby(geography)[[mode, population]]:
        quant = weighted_quantiles_interpolate(group[mode].values, group[population].values, quantiles = quantile)

        id_list.append(name)
        pov_list.append(float(quant))

    # create data frame of percentiles
    rdf = pd.DataFrame({geography : id_list, 'pov' : pov_list})
    # rdf['pov'] = rdf.pov.astype(float)
    return rdf

def create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = 0.25, pov_geography = False):
    if not pov_geography:
        pov_geography = group_column

    # calculate per geography pov
    pov = pov_line(gdf, pov_geography, access_column, population_column, quantile)
    # calculate demographic totals per geography
    n = gdf.groupby(group_column)[demographics].sum()
    n = n.T.copy()

    # add geography pov line
    pdf = gdf.merge(pov, on = pov_geography, how = 'left')

    # calculate delta from pov line
    pdf["_delta"] = pdf['pov'] - pdf[access_column]
    # calculation is only on those in poverty
    pdf = pdf[pdf["_delta"] > 0].copy()
    # recalculate delta as % of pov line
    pdf["_delta"] = pdf["_delta"] / pdf['pov']
    # raise delta to FGT power
    pdf["_delta"] = pdf["_delta"].pow(alpha)

    # multiple all demographics values by delta
    pdf[demographics] = pdf[demographics].multiply(pdf['_delta'], axis="index")

    # sum demo-deltas by geography
    totals = pdf.groupby(group_column)[demographics].sum()

    # format tables
    totals = totals.T.copy()
    totals.columns = [str(s) + '_count' for s in totals.columns]
    n.columns = [str(s) + '_n' for s in n.columns]

    comb = pd.concat([totals, n], axis=1)

    idxs = pdf[group_column].unique()

    # calculate percentages
    for idx in idxs:
        comb[idx] = comb[f'{idx}_count'] / comb[f'{idx}_n']

    comb = comb[idxs]

    return comb

### Import Data

# group_column = 'CSD_UID'
# access_column = 'rf_ai_cyc_csd'
# alpha = 0

if __name__ == "__main__":

    gdf = pd.read_csv('TransportEquityDashboardDAs.csv', dtype = {'CD_UID' : str, 'CMA_UID' : str, 'CSD_UID' : str})
    gdf['N_UID'] = '1'

    population_column = 'Population'

    ## Run Calculations

    # creates columns with the name 'fgt_{demographic}_{destination}_{mode}'
    for geography in geographies:
        df_list = []
        group_column = geography.upper() + '_UID' if geography != '' else 'N_UID'
        geography = '_' + geography if geography != '' else geography

        for destination in destinations:
            for mode in modes:
                # access column construction - {destination}_ai_{mode} + _{geography} if not national
                # access_column = f'{destination}_ai_{mode}{geography}'
                access_column = column_format(destination, mode, geography)
                tab = create_fgt(gdf, group_column, access_column, population_column, alpha = 0)
                tab = tab.T.copy()
                tab.columns = ['fgt_' + '_'.join(s.split('_')[:-1]) + f'_{destination}_{mode}' for s in tab.columns]
                df_list.append(tab)
        full_geoid = pd.concat(df_list, axis = 1)
        full_geoid.reset_index(names = group_column, inplace = True)
        full_geoid.to_csv(f'output/{group_column}.csv', index = False)

## POV level changes

    geography = ''
    destination = 'emp'
    mode = 'cyc'
    group_column = 'N_UID'
    access_column = f'{destination}_ai_{mode}{geography}'

    national = create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = .25)
    national.rename(columns = {'1' : 'national'}, inplace = True)

    cma = create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = .25, pov_geography = 'CMA_UID')
    cma.rename(columns = {'1' : 'cma'}, inplace = True)

    csd = create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = .25, pov_geography = 'CSD_UID')
    csd.rename(columns = {'1' : 'csd'}, inplace = True)

    cd = create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = .25, pov_geography = 'CD_UID')
    cd.rename(columns = {'1' : 'cd'}, inplace = True)

    fig, ax = plt.subplots()
    national.plot(ax = ax)
    cma.plot(ax = ax)
    csd.plot(ax = ax)
    cd.plot(ax = ax)
    plt.show()

    full = pd.concat([national, cma, csd, cd], axis = 1)

    ### Threshold Test

    geography = ''
    destination = 'emp'
    mode = 'cyc'
    group_column = 'N_UID'
    access_column = f'{destination}_ai_{mode}{geography}'

    qdfs = []
    for quant in range(1, 20):
        qs = create_fgt(gdf, group_column, access_column, population_column, alpha = 0, quantile = quant / 20)
        qs.rename(columns = {'1' : str(quant / 20)}, inplace = True)
        qdfs.append(qs)

    qdf = pd.concat(qdfs, axis = 1)

    ## Plot Thresholds

    import matplotlib.pyplot as plt

    qdf = qdf.T.copy()
    qdf.reset_index(inplace = True, names = ['quant'])
    qdf.quant = qdf.quant.astype(float)

    fig, ax = plt.subplots()

    for col in qdf.columns:
        if col != 'quant':
            qdf[col] = qdf[col] / qdf['quant']
            ax.plot(qdf.quant, qdf[col], label = col)
        else:
            pass

    plt.suptitle('Impact of Threshold Value on National FGT0 Scores')

    ax.set_ylabel('Ratio of Expected Value')
    ax.set_xlabel('Threshold Quantile')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
