# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import copy
import matplotlib.pyplot as plt
import os
from sys import argv
from locale import atof, atoi


"""# config"""
kwarg = {'rho_12':0.41,
         'rho_13':0.72,
         'rho_23':0.82,
         'start_mean': 1500,
         'update_var':False}

init_sig = [50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150, 160, 170, 175, 180, 200, 205, 210]
# init_sig = [50, 60, 80, 100, 120, 150, 180, 200, 220, 250, 280, 300]

"""### encode Function"""

def encode_marks(marks):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(marks)
    oh = np.zeros((len(marks), len(encoder.classes_)))

    oh[np.arange(len(marks)), encoded] = 1

    return oh, encoder.classes_


"""### Data input and split"""

data = pd.read_csv('single.csv') #load load

data['tourney_date'] = pd.to_datetime(data.tourney_date) #change data time to datetime
data['year'] = data['tourney_date'].dt.year

Train = data[data['tourney_date'].dt.year <= 2017] 
Test = data[data['tourney_date'].dt.year > 2017] 

Train = Train.reset_index()
Test = Test.reset_index() #Reset the index
winners, losers = Train['winner_id'], Train['loser_id'] 
winners_val, losers_val = Test['winner_id'], Test['loser_id']

surface = encode_marks(Train['surface'])[0] #Encode the surfaces
#Clay[1, 0, 0]
#Grass[0, 1, 0]
#Hard[0, 0, 1]

surface_val = encode_marks(Test['surface'])[0]
# data = data.reset_index()

"""# input arguments 
par1: lower bound on variance, e.g. par1 = 0, 50, 60, with par1 = 0 corresponds to origial formula
par2: setting of rho, par2 == 0/1 corresponds to models with/without correlation factor
par3: additional factor controling variance reduction, e.g. par3 = 2, 5, with par3 = 1 corresponds to original formula
"""

par1 = argv[1]
par1_i = atoi(par1)
par2 = argv[2]
par2_i = atoi(par2)
par3 = argv[3]
par3_i = atoi(par3)

folder =  'OUTNEWsingles_b'+par1+'_r'+par2+'_d'+par3
folderS = folder+'/'

print(folder)
print(folderS)

try:
  os.mkdir(folderS)
except:
  pass

os.chdir(folder)


if par2_i ==1:
   kwarg['rho_12']=1
   kwarg['rho_13']=1
   kwarg['rho_23']=1


"""### Exploratory Data Analysis"""


"""### more Functions"""

def sigmoid(mu_w, mu_l):
  b = np.log(10)/400 
  x = b*(mu_w-mu_l)
  return 1/(1+np.exp(-x))

def calculate_ratings(winners, losers, surface, var_1, var_2, var_3, rho_12, rho_13, rho_23, start_mean, file_name = None, update_var = True, mode = 'Train', prior_mean = None, prior_var = None):
  if file_name==None: file_name = 'singles_'+ mode + ('_both.txt' if update_var else '_only.txt')
  if mode.lower() =='train': 
    prior_mean = defaultdict(lambda :(np.array([[start_mean],[start_mean],[start_mean]],dtype = 'float64')))
    prior_var = defaultdict(lambda :(np.array([[var_1],[var_2],[var_3]],dtype = 'float64')))
  elif mode.lower() not in ['test','train']: 
    print(f'WARNING: Ur input of mode is {mode} and the current mode is "Train" ')

  with open(file_name,'w',encoding="utf-8") as f:
    str2 = 'game, current_winner, current_loser, winner_mean_old, loser_mean_old,  winner_mean, loser_mean, winner_std, loser_std, phat'
    f.write(str2+'\n')   
    count,tie,total,total_discrepancy = 0,0,0,0
    b= np.log(10)/400
    
    for game in np.arange(winners.shape[0]):
        current_winner = winners[game]
        current_loser = losers[game]
        # current surface [1,3]
        current_surface = surface[game]
        # current player
        current_player = [current_winner, current_loser]
        for player in current_player:
            if player not in prior_mean:
                prior_mean[player]
                prior_var[player]
        
        new_mean = prior_mean.copy()
        new_var = prior_var.copy()
        # winners & loser rating
        winner_mean = new_mean[current_winner]
        loser_mean = new_mean[current_loser]
        # 
        winner_var = new_var[current_winner]
        loser_var = new_var[current_loser]
        
        # calculate p(x)
        mu_w = current_surface.dot(winner_mean)[0] # float
        mu_l = current_surface.dot(loser_mean)[0] #float

        p_w = sigmoid(mu_w, mu_l) # float
        p_l = 1-p_w
 
        # compare and count the correct prediction
        if mu_w > mu_l:
          count +=1
        if mu_w == mu_l:
          tie +=1
        total +=1
        
        # calculate discrepancy
        discrepancy = -np.log(p_w)
        total_discrepancy += discrepancy
        
        #Calculate C
        C = 1/(1 + (b**2)*p_w*p_l*(current_surface.dot(winner_var) + current_surface.dot(loser_var)))
        #Calculate K
        rho_mat = np.array([[1, rho_12, rho_13],
                  [rho_12, 1, rho_23],
                  [rho_13, rho_23, 1]])        
        S_wi = np.sqrt(current_surface.dot(winner_var))
        S_li = np.sqrt(current_surface.dot(loser_var))
        cur_rho = current_surface.dot(rho_mat)
        K_w = S_wi*np.sqrt(winner_var).T*cur_rho.T*b*C
        K_l = S_li*np.sqrt(loser_var).T*cur_rho.T*b*C
        
        # update player mean
        winner_mean_new = winner_mean + K_w.T*(1-p_w)
        loser_mean_new = loser_mean + K_l.T*(-p_l)
        new_mean[current_winner] = winner_mean_new
        new_mean[current_loser] = loser_mean_new
        prior_mean = new_mean

        #Calculate new win prob and new C
        mu_w_new = current_surface.dot(winner_mean_new)[0] # float
        mu_l_new = current_surface.dot(loser_mean_new)[0] #float
        p_w_new = sigmoid(mu_w_new, mu_l_new)
        p_l_new = 1-p_w_new
        C_new = 1/(1 + (b**2)*p_w_new*p_l_new*(current_surface.dot(winner_var) + current_surface.dot(loser_var)))
        #Calculate L, L_w, L_l
        L = np.square(cur_rho)
        L_w = (b**2)*p_w_new*p_l_new*L*winner_var.T*C_new
        L_w = L_w/par3_i
        L_l = (b**2)*p_w_new*p_l_new*L*loser_var.T*C_new
        L_l = L_l/par3_i
        # update player variance
        if update_var:
          tmp_w = winner_var*(1-L_w).T
          tmp_l = loser_var*(1-L_l).T
          new_var[current_winner] = np.array([[max(np.array([par1_i**2]),tmp_w[0])[0]],[max(np.array([par1_i**2]),tmp_w[1])[0]],[max(np.array([par1_i**2]),tmp_w[2])[0]]],dtype = 'float64')
          new_var[current_loser] = np.array([[max(np.array([par1_i**2]),tmp_l[0])[0]],[max(np.array([par1_i**2]),tmp_l[1])[0]],[max(np.array([par1_i**2]),tmp_l[2])[0]]],dtype = 'float64')         
          # new_var[current_winner] = winner_var*(1-L_w).T
          # new_var[current_loser] = loser_var*(1-L_l).T
          prior_var = new_var
        

        var_w_new = current_surface.dot(winner_var)[0] # float
        var_l_new = current_surface.dot(loser_var)[0] #float
   
        
        # save record
        str2 ='%2d, %s, %s, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f\n'%(game+1,current_winner,current_loser,mu_w,mu_l, mu_w_new, mu_l_new, np.sqrt(var_w_new), np.sqrt(var_l_new), p_w)
        f.write(str2) # save
        
    f.close()
    error = total-count-tie
    acc = count/(total-tie)
    return prior_mean, prior_var, acc, (total,count,tie), total_discrepancy/winners.shape[0] # total(n) - (accurate(a) + ties(t)) = error (e)     
   

"""### Different Sigma0"""

print(kwarg)

file_name = 'accRate_singles'
file_name +='_both.txt' if kwarg['update_var'] else '_only.txt'
print(file_name)

accTr=[]
accTe=[]
num=[]
discrepancy = []
with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s, %2s, %2s, %2s\n'%('sig','accTrain','accTest',"(n,a,t)",'discrepancy'))
  for i in range(0, len(init_sig)):
     # print('sigma0 = ',init_sig[i])
     est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Train') 
     _, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var) 
     #print('update mu only and use surface',f'acc_rate:{np.round(acc_te, 4)}')
     accTr.append(acc_tr)
     accTe.append(acc_te) 
     num.append(num_te)
     discrepancy.append(discrep)
     str_w = f'{init_sig[i]:3d}, {acc_tr:.4f},{acc_te:.4f},{num_te},{discrep:.4f}'
     f.write(str_w+"\n") # save
f.close()   


acc_idx_tr_only = accTr.index(max(accTr))
acc_idx_te_only = accTe.index(max(accTe))
print('sigma with best accuracy in train: %d'%(init_sig[acc_idx_tr_only]))
print('sigma with best accuracy in test: %d'%(init_sig[acc_idx_te_only]))
disc_idx_tr_only = discrepancy.index(min(discrepancy))
print('sigma with smallest discrepancy in train: %d'%(init_sig[disc_idx_tr_only]))

est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, **kwarg, mode = 'Train')
_, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var)


kwarg['update_var']=True
print(kwarg)
file_name = 'accRate_singles'
file_name +='_both.txt' if kwarg['update_var'] else '_only.txt'
print(file_name)

accTr1=[]
accTe1=[]
num1=[]
discrepancy1 = []
with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s, %2s, %2s, %2s\n'%('sig','accTrain','accTest',"(n,a,t)",'discrepancy'))
  for i in range(0, len(init_sig)):
     #print('sigma0 = ',init_sig[i])
     est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Train') 
     _, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var) 
     #print('update mu & sigma and use surface',f'acc_rate:{np.round(acc_te, 4)}')
     accTr1.append(acc_tr)
     accTe1.append(acc_te) 
     num1.append(num_te)
     discrepancy1.append(discrep)
     str_w = f'{init_sig[i]:3d}, {acc_tr:.4f},{acc_te:.4f},{num_te},{discrep:.4f}'
     f.write(str_w+"\n") # save
f.close()

acc1_idx_tr_only = accTr1.index(max(accTr1))
acc1_idx_te_only = accTe1.index(max(accTe1))
print('sigma with best accuracy in train: %d'%(init_sig[acc1_idx_tr_only]))
print('sigma with best accuracy in test: %d'%(init_sig[acc1_idx_te_only]))
disc1_idx_tr_only = discrepancy1.index(min(discrepancy1))
print('sigma with smallest discrepancy in train: %d'%(init_sig[disc1_idx_tr_only]))

est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, **kwarg, mode = 'Train')
_, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var)


file_name = 'errRate'
file_name +='.txt'
# file_name = 'errRate.txt'
with open(file_name, 'w', encoding="utf-8") as f:  
  f.write('%3s\n'%('mean only'))
  f.write('%3s, %2s,%2s\n'%('sig','accTest','discrepancy'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe[i]:.4f} & {discrepancy[i]:.4f}\\\\'
     f.write(str_w+"\n") # save     
        
  f.write("\n")     
  f.write('%3s\n'%('both mu and sigma'))
  f.write('%3s, %2s,%2s\n'%('sig','accTest','discrepancy'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe1[i]:.4f} & {discrepancy1[i]:.4f}\\\\'
     f.write(str_w+"\n") # save  
        
  f.write("\n")     
  f.write('%3s\n'%('together'))
  f.write('%3s, %2s,%2s, %3s, %2s,%2s\n'%('sig','accTest','discrepancy','sig','accTest','discrepancy'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe[i]:.4f} & {discrepancy[i]:.4f} & & & {init_sig[i]:3d} & {accTe1[i]:.4f} & {discrepancy1[i]:.4f}\\\\'
     f.write(str_w+"\n") # save       
f.close()   
