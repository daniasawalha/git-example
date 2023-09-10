# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:28:03 2022

@author: Dania
"""
import pandas as pd
from faker import Faker
import csv
output=open('data.csv','w')
fake=Faker()
header=['name','age','street','city','state','zip','lat']
mywirter=csv.writer(output)
mywirter.writerow(header)
for i in range(1000):
    mywirter.writerow([fake.name(),fake.random_int(min=18,max=80,step=1),fake.street_address(),fake.city(),fake.state(),fake.zipcode(),fake.longitude(),fake.latitude()])
output.close()

number=0
count=0
num=0
m=40
mm=40
sumiation=0
with open('data.csv') as f:
    myreader=csv.DictReader(f)#,quoting=csv.QUOTE_NONNUMERIC(['age']))
    #s=csv.QUOTE_NONNUMERIC()
    header=next(myreader)
    for row in (myreader):
        
        #sumiation=sum(row['age'])
        pd.
        count +=1
        if float(row['age'])>m:
            m=float(row['age'])
        elif float(row['age'])<mm:
            mm=float(row['age'])
        #maximum=max(float(row['age']))
        #minimum=min(row['age'])
        sumiation +=float(row['age'])
    #for row in myreader:  
        if(float(row['age'])>40):
            number +=1
        elif (float(row['age'])<20):
            num +=1
        avrage=sumiation/count
       # df = pd.DataFrame(myreader)
    #df['age'] = df['age'].astype(int)  
print("max age:(",m,") the min age:(",mm,") the avg of ages:(",avrage,") number of people their age more than 40:(",number,") number of people their age less than 20:(",num,")")