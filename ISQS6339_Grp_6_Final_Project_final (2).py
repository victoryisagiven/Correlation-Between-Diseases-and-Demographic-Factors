# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:04:42 2023

Dwayne Hoelscher
Casey Hicks
Victoria Stoner
Hoshana Maharjan
"""

# =============================================================================
# #%% import modules
# =============================================================================
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# file path variable, change as needed
# =============================================================================
VAR_FILE_PATH = r'C:\\Users\\dwayn\\OneDrive - Texas Tech University\\Classes\\ISQS 6339 Business Intelligence\\Group6\\Final_Project\\Final_Datasets\\'

# =============================================================================
# file name variables and URL for source, change as needed
# =============================================================================

VAR_SDOH = 'SDOH.xlsx'   ### https://www.ahrq.gov/downloads/sdoh/sdoh_2020_tract_1_0.xlsx
VAR_DX = 'Diagnosis_State_County.csv'   ### https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/duw2-7jbt
VAR_ETOH = 'Liquor_Establishments.csv'  ### https://data.census.gov/table?q=CBP2020.CB2000CBP&n=4453&tid=CBP2020.CB2000CBP

# SDOH takes a bit to run for Dwayne

# =============================================================================
# Get the data from the spreadsheets (excel and csv)
# =============================================================================

df_sdoh_data = pd.read_excel(VAR_FILE_PATH+VAR_SDOH,sheet_name='Data')
df_dx = pd.read_csv(VAR_FILE_PATH+VAR_DX)
df_etoh = pd.read_csv(VAR_FILE_PATH+VAR_ETOH,header=1)

# =============================================================================
# combine the data into a working dataframe
# get the initial county and state information from the SDOH dataset
# This SDOH dataset is the source of truth for our data and will only include
# states in the final dataset (no territories)
# =============================================================================

df_wrk = df_sdoh_data[['STATE','COUNTY','STATEFIPS','COUNTYFIPS']].copy().drop_duplicates()

# =============================================================================
# merge diagnosis data and flatten the diagnosis data including mean values by county
# =============================================================================

df_dx_pivot = df_dx[
    (df_dx['DataValueTypeID']=='CrdPrv')
    & ((df_dx['MeasureId']=='DIABETES')
    |(df_dx['MeasureId']=='CANCER')
    |(df_dx['MeasureId']=='DEPRESSION'))
    ].pivot_table(
    index = 'LocationID' #,'LocationName'] #['MeasureId','Short_Question_Text']
    , columns = 'MeasureId'
    , values = 'Data_Value'
    , aggfunc = 'mean'
    ).reset_index()
df_wrk = df_wrk.merge(df_dx_pivot, how='left', left_on='COUNTYFIPS', right_on='LocationID')
#drop location id and use the countyfips
df_wrk = df_wrk.drop('LocationID',axis=1)

# check the total by state and district of columbia
df_wrk_dx_gb_state = df_wrk.groupby(['STATEFIPS','STATE']).agg({'COUNTY':'count','DIABETES':'sum','CANCER':'sum','DEPRESSION':'sum'})
df_wrk_dx_gb_state = df_wrk_dx_gb_state.rename(columns={'COUNTY':'county_count','DIABETES':'diabetes_sum','CANCER':'cancer_sum','DEPRESSION':'depression_sum'})

# drop the territories listed in the COUNTYFIPS and those outside the United States and Washington DC
df_wrk = df_wrk[df_wrk['STATEFIPS']<60]

# =============================================================================
# Merge the SDOH data with the internet data and the population
# =============================================================================

df_wrk = df_wrk.merge(df_sdoh_data.groupby(['COUNTYFIPS'])[['ACS_PCT_HH_NO_INTERNET','ACS_PCT_FEMALE','ACS_PCT_MALE','ACS_MEDIAN_HH_INC']].mean().reset_index()
                      , on = ['COUNTYFIPS'], how = 'inner')

df_wrk = df_wrk.merge(df_sdoh_data.groupby(['COUNTYFIPS'])[['ACS_TOT_CIVIL_POP_ABOVE18']].sum().reset_index()
                      , on = ['COUNTYFIPS'], how = 'inner')

# =============================================================================
# merge alcohol data (ETOH = Ethyl Alcohol)
# first split the geographic identifier code to get the countyfips to allow for joins
# rename the Employment size of establishments code to ESTABLISMENT_CODE 
# rename the Number of establishments to ESTABLISHMENT_NUMBER
# =============================================================================

df_etoh[['geoid','COUNTYFIPS']] = df_etoh['Geographic identifier code'].str.split('US',expand=True)
df_etoh['COUNTYFIPS'] = df_etoh['COUNTYFIPS'].str.lstrip('0')
df_etoh['COUNTYFIPS'] = df_etoh['COUNTYFIPS'].astype(int)

df_etoh = df_etoh.rename(columns={'Employment size of establishments code':'ESTABLISMENT_CODE','Number of establishments':'ESTABLISHMENT_TOTAL'})

df_wrk = df_wrk.merge(df_etoh[df_etoh['ESTABLISMENT_CODE']==1].groupby(['COUNTYFIPS'])['ESTABLISHMENT_TOTAL'].sum().reset_index()
                        , how = 'left', on = ['COUNTYFIPS'])

# =============================================================================
# impute the number of liquor stores for missing counties by population
# and total liquor establishments
# imputed numbers in the ESTABLISHMENT_TOTAL field, original = integers, imputed = float
# =============================================================================

df_wrk['ESTABLISHMENT_TOTAL'] = KNNImputer(n_neighbors=3).fit_transform(df_wrk[['ACS_TOT_CIVIL_POP_ABOVE18','ESTABLISHMENT_TOTAL']])[:, 1] # impute establishments
df_wrk['POPULATION_PER_ESTABLISHMENTS'] = df_wrk['ACS_TOT_CIVIL_POP_ABOVE18']/df_wrk['ESTABLISHMENT_TOTAL']

df_wrk = df_wrk.drop(columns=['ESTABLISHMENT_TOTAL'],axis=1)

# =============================================================================
# Investigate the initial descriptive statistics including corrlation
# =============================================================================

df_wrk_columns_list = df_wrk.dtypes.reset_index()
df_wrk_columns_list = df_wrk_columns_list[df_wrk_columns_list[0]=='float64'].values[:,0].tolist()

df_wrk_describe = df_wrk.describe().T

df_wrk_describe = df_wrk_describe.merge(df_wrk[df_wrk_columns_list].agg(['median','var']).T,left_index=True, right_index=True,how='inner')

df_wrk_corr = df_wrk[df_wrk_columns_list].corr()

sns.set(style='white')

## =============================================================================
## graph results using kernal density in the lower portion, histograms in the diagonals,
## and scatter plots incluting coefficient lines with r values and p values where 
## *** <- 0.001, ** <= 0.01, * <= 0.05
## https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
## =============================================================================

def corrfunc(x, y, **kws):
  r, p = stats.pearsonr(x, y)
  p_stars = ''
  if p <= 0.05:
    p_stars = '*'
  if p <= 0.01:
    p_stars = '**'
  if p <= 0.001:
    p_stars = '***'
  ax = plt.gca()
  ax.annotate('r = {:.2f} '.format(r) + p_stars,
              xy=(0.5, 0.9), xycoords=ax.transAxes)

def annotate_colname(x, **kws):
  ax = plt.gca()
  ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
              fontweight='bold')
  
def cor_matrix(df):
  g = sns.PairGrid(df, height=5)
  g.map_upper(sns.regplot, scatter_kws={'s':10}, line_kws={"color": "red"})
  g.map_diag(sns.histplot, kde=True, kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
  g.map_diag(annotate_colname)
  g.map_lower(sns.kdeplot, cmap='Blues_d')
  g.map_upper(corrfunc)

  # Remove axis labels, as they're in the diagonals.
  for ax in g.axes.flatten():
    ax.set_ylabel('')
    ax.set_xlabel('')
  return g

cor_matrix(df_wrk[df_wrk_columns_list].dropna(how='any'))