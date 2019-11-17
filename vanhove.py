#!/bin/env python3
#################################################################
# This uses MDAnalysis module. You have to have this module to  #
# run this progrm. The coordinate/velocity parsing is done by   #
# MDAnalysis module. This module will plot scatting function    #
# S(q,t). You can also output Gs(r,t) self-part of Van Hove     # 
# Correlation funciton.                                         #
#################################################################
import MDAnalysis as mda 
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

print("loading trajectory")
top   =  'afeq.tpr'
traj  =  'afeq.xtc'
print("loading is Done")
start =  0
end   =  100
step  =  1
tau   =  1
dr    =  1
u     =  mda.Universe(top,traj)
sel   =  'resname DLPC and name C3B'
dt    =  u.trajectory.dt

def dist_r(r):
        dist = np.sum((r[:,:])**2)
        return dist 

def real_disp(dr,box):
       rul = [box[i]*0.5 for i in range(3)]
       for i in range(3):
           for j in range(len(dr)):
               if dr[j,i] > rul[i]:
                  dr[j,i] = dr[j,i] - box[i]
               if dr[j,i] < (-rul[i]):
                  dr[j,i] = dr[j,i] + box[i]
       return np.array(dr)

def vanhove_t(r,t,density=True):
    temp=[]
    if t != 0:
        for f in range(0,len(r)):
            temp_data = r[f][:-t,] - r[f][t:,:]   
            temp.append(np.linalg.norm(temp_data,axis=1))
    else:
        for i in range(len(r)):
            temp.append(np.linalg.norm(r[i],axis=1))
    temp = np.concatenate(np.array(temp))
    return temp

def get_plots(dr,t):
     temp = vanhove_t(dr,t,True)
     freq,bins = np.histogram(temp,bins=np.linspace(0,t*5+50,100),density=density)
     return plt.plot(bins[:-1]+(0.5*(bins[1]-bins[0])),freq,'-',label='t = '+str(int(dt)*t)+'ps')

def s(q,t,dr):
     temp = vanhove_t(dr,t)
     freq,bins = np.histogram(temp,bins=np.linspace(0,t*5+50,100),density=True)
     s = [freq[i]*np.cos(q*bins[i]) for i in range(len(freq))]
     s = np.sum(np.asarray(s))
     return s

def main():
     mpl.style.use('classic')
     plt.rc('figure',facecolor='white')     
     nf = len(u.trajectory[start:end:step])      
     dr = []
     
     selection = u.select_atoms(sel)
     
     print("fixing the trajectory displacement")
     
     for fr in range(nf):
         print("frame",fr)
         u.trajectory[fr]
         box = u.dimensions[0:3] 
         if fr == 0:
             prev = selection.positions
         else:
             curr = selection.positions
             vect = curr-prev
             disp = real_disp(vect,box)
             dr.append(disp)
             prev = curr
     dr = np.array(dr)
 #    for t in range(1,40,10):
  #       get_plots(dr,t)         
  #   plt.legend(loc='upper right')
  #   plt.ylim(0,0.15)
  #   plt.xlim(0,60)
  #   plt.show()
     Sqt={}
     for q in np.linspace(0.018,0.135,14):
           temp_sqt = [] 
           for t in range(0,1000,5):
                temp_sqt.append(s(q,t,dr))
           Sqt[q]=temp_sqt
     for k,v in Sqt.items():
         plt.plot([(i*dt/1000) for i in range(0,1000,5)],v/v[0],linestyle='--',marker='o',label="q="+str("%.3f" % k))
     plt.legend(loc="upper right")
     plt.xlabel('Time(ns)')
     plt.show()

if __name__ == '__main__':
      main()
