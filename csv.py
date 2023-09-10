# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:46:59 2022

@author: Dania
"""

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 85)
pd.set_option('display.max_columns', 6)
per=pd.read_excel(("GDPpercapita.xlsx"),
    sheet_name="OECD.Stat export",
    skiprows=4,
    skipfooter=1,
    usecols="A,C:T")
per.head()   
per.info()   

per.rename(columns={'Year':'metro'},inplace=True)  
per.metro.str.startswith(' ').any()   
per.metro.str.endswith(' ').any() 
per.metro=per.metro.str.strip()
for col in per.columns[1:]:
    per[col]=pd.to_numeric(per[col],errors='coerce')
    per.rename(columns={col:'pcGDP'+col},inplace=True)
per.head()
per.dtypes
per.describe()
per.dropna(subset=per.columns[1:],how='all',inplace=True)
per.shape
per.set_index('metro',inplace=(True))