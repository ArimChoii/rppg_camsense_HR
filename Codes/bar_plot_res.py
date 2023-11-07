#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:54:21 2020

@author: zahid
"""
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('PS')
# plt.style.use('ggplot')

#%% Real Deal Starts Here
import numpy as np

N = 2
tot_peak = (72,72)
TP = np.array([63, 68])/72*100
FP = np.array([8, 6])/72*100
FN = np.array([9, 4])/72*100

fig = plt.figure()
ax2 = fig.add_subplot(122)

ind = np.arange(N)      
width = 0.5    
plt.bar(ind, TP, width/4, label='TP peaks', hatch ='/')
plt.bar(ind + width/3, FP, width/4, label='FP', hatch ='-')
plt.bar(ind + 2*width/3, FN, width/4, label='FN', hatch ='x')

plt.yticks(fontsize=50)

# plt.ylabel('Scores')
plt.title('(b) UBFC-RPPG dataset', fontsize=35, fontweight="bold")

#plt.xlabel("UBFC-rPPG Dataset", fontsize=35, fontweight="bold")


xlabels = ( 'Tx from \n person', 'Tx from \n MTL')

plt.xticks(ind + width/3, xlabels)
ax2.set_xticklabels(xlabels, rotation=45, fontsize=35)

# plt.legend(loc='best', ncol=3, fontsize=30)

plt.savefig('bar_chart_sample1.svg', format = 'svg', dpi= 500, bbox_inches="tight")

plt.show()

#%%
N = 2
tot_peak = (72,72)
TP = np.array([63, 68])/72*100
FP = np.array([8, 6])/72*100
FN = np.array([9, 4])/72*100

ax2 = fig.add_subplot(122)

ind = np.arange(N)      
width = 0.5    
plt.barh(ind, TP, width/4, label='TP peaks')
plt.barh(ind + width/3, FP, width/4, label='FP')
plt.barh(ind + 2*width/3, FN, width/4, label='FN')

plt.xticks(fontsize=30)
plt.xlabel('Percentage', fontweight="bold", fontsize=40)

# plt.ylabel('Scores')
plt.title('UBFC-RPPG dataset', fontsize=40, fontweight="bold")

# plt.ylabel("UBFC-rPPG Dataset", fontsize=40, fontweight="bold")

xlabels = ( 'Tx from \n person', 'Tx from \n MTL')

plt.yticks(ind + width/2, xlabels)
ax2.set_yticklabels(xlabels, rotation=90, fontsize=35)

plt.legend(loc='upper right', ncol=1, fontsize=30)

# plt.savefig('bar_chart_sample1.eps', format = 'eps', dpi= 500, bbox_inches="tight")

plt.show()

#%% Scatter plot 
fig = plt.figure(figsize=(19.20,10.80))
epoch =  [86, 86, 86]

per_tr = [0.07, 0.08, 0.076]
per_val = [0.12, .09, 0.11]

mark_siz = 1300

plt.scatter(epoch, per_tr, c = 'blue', marker = 'o', s = mark_siz, label = 'Personalized Train')
plt.scatter(epoch, per_val, c = 'red', marker = 'o', s = mark_siz)


plt.xticks(fontsize=35)
plt.yticks(fontsize=35)


epoch =  [16, 16, 16]
tr_tr = [0.06, 0.065, 0.05]
tr_val = [0.09, .091, 0.092]

plt.scatter(epoch, tr_tr, c = 'blue', marker = '^', s = mark_siz, label = 'MTL')
plt.scatter(epoch, tr_val, c = 'red', marker = '^', s = mark_siz)



epoch =  [95, 95, 95]
mtl_tr = [0.08, 0.085, 0.09]
mtl_val = [0.1, .15, 0.12]

plt.scatter(epoch, mtl_tr, c = 'blue',  marker = 'x', s = mark_siz, label = 'Tx Learning')
plt.scatter(epoch, mtl_val, c = 'red',  marker = 'x', s = mark_siz)


plt.xlabel('No of Epochs', fontweight="bold", fontsize=40)
plt.ylabel('MSE', fontweight="bold", fontsize=40)
plt.title('Validation vs Training MSE',fontsize=40, fontweight="bold")

plt.legend(loc='best', ncol=1, fontsize=35)

plt.savefig('tr_val_MSE.eps', bbox_inches="tight", format = 'eps', dpi= 500)

#%%
fig = plt.figure(figsize=(19.20,10.80))
# create data
# Setting size in Chart based on 
# given values
sizes = [100, 500, 70, 54, 440]
  
# Setting labels for items in Chart
labels = ['Apple', 'Banana', 'Mango', 'Grapes', 'Orange']
  
# colors
colors = ['#FF0000', '#0000FF', '#FFFF00', '#ADFF2F', '#FFA500']
  
# explosion

explode = (0.05, 0.05, 0.05, 0.05, 0.05)
  
# Pie Chart
plt.pie(sizes, colors=colors, labels=labels,
        autopct='%1.1f%%', pctdistance=0.8, 
        explode=explode, textprops={'fontsize': 25}) # explode for open version
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Favourite Fruit Survey')
  
# Add Legends
plt.legend(labels, loc="upper right")
  
# Displaing Chart
plt.show()