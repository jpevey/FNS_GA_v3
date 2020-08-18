#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:56:43 2020

@author: CameronSalyer
"""
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import csv
import os

def pull_fluxspect(file):
    inputfile = open(file, 'r')
    bins = []
    current_vals= []
    current_unc = []
    total = []
    total_unc = []
    i = 0
    for line in inputfile:
        if line == '      energy   \n':
            p = 0
            for n in range(i, i+253):
                lines = inputfile.readline(n)
                split_l = lines.split(' ')
                length = len(split_l)
                bins.append(split_l[length-5])
                #print(split_l[length-2])
                current_vals.append(np.float64(split_l[length-2]))
                current_unc.append(np.float64(split_l[length-1]))
                #ph.append(placeholder)
                if p == 252:
                    total.append(split_l[len(split_l)-2])
                    total_unc.append(split_l[len(split_l)-1])
                p = p + 1
               
            current_vals.pop(252) #extracting last value ('total flux')
            current_unc.pop(252)
        i = i + 1
    #print(bins, len(bins))
    bins.pop(252)
    #allvals.append(current_vals)
    #allunc.append(current_unc)
    #print(current_vals)
    return current_vals, current_unc
def pulldata():
    #READ in SFR Flux Data for Spectrum comparisons
    workbook = xlrd.open_workbook("SFR_Flux_Data.xlsx")
    sheet = workbook.sheet_by_index(2)
    bins = []
    nums = []
    sfrflux = []
    sfrflux_unc = []
    i=0
    for rowx in range(sheet.nrows):
        if i == 0:
            pass
        else :
            values = sheet.row_values(rowx)
            nums.append(values)
        i = i + 1
    for val in nums:
        bins.append(val[0])
        sfrflux.append(val[1])
        sfrflux_unc.append(val[2])

    
    #Read in computed Representativity and Flux values
    repvals = []
    case = []
    mats = []
    with open('_output.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:               
            #print(row)
            #if row[0] == 'fuel_index_multiplier':
                #print(line_count)
            if line_count > 66:
                #print(row[2])
                repvals.append(float(row[4]))
                case.append(row[2])
                mats.append(row[9:(len(row)-1)])

            line_count += 1
    full_list = repvals, case, mats           
    max_val = max(repvals)
    #print(max_val)
    #print(pull_fluxspect(case[i]))
    i = 0
    p = 0
    for line in repvals:
        if line == max_val and p < 1:
            #print(case[i])
            best_rep = line
            best_material = mats[i]
            #print(best_material)
            bestcase_flux, unc = pull_fluxspect(str(case[i]+'o'))
            p = p + 1
        i = i + 1
    #print(total_list)
    return repvals, bestcase_flux, best_rep, best_material, bins, sfrflux



     
def Plotting():
    #PLOT ALL REPS ans show it getting better
    #PLOT best-worst Spectras vs SFR
    repvals, best_flux, best_rep, bestmat, bins, sfrflux = pulldata()
    best_plot = []
    sfr_plot = []
    i=0
    for fl in best_flux:
        if i == 0:
            best_plot.append(fl/(np.log10(bins[i])))
            i += 1
        else:
            best_plot.append(fl/(np.log10(bins[i]/bins[i-1])))
            i += 1
    i=0
    for sfr in sfrflux:
        if i == 0:
            sfr_plot.append(sfr/(np.log10(bins[i])))
            i += 1
        else:
            sfr_plot.append(sfr/(np.log10(bins[i]/bins[i-1])))
            i += 1
    bestmat = []
    for val in best_mat:
        val = int(val)
        if val == 1:
            mat = str('V')
            bestmat.append(mat)
        elif val == 2:
            mat = str('P')
            bestmat.append(mat)
        elif val == 3:
            mat = str('F')
            bestmat.append(mat)
        elif val == 4:
            mat = str('S')
            bestmat.append(mat)
            
    bestrep = str(best_rep)
    bestmat_str = str(bestmat)
    #correcting for division by zero
    np.seterr(divide = 'ignore') 
    plt.scatter(np.log10(bins),np.log10(best_plot), label=str('Best Case:'+bestmat_str), s=10)
    plt.scatter(np.log10(bins),np.log10(sfr_plot), label='SFR', s=10)
    #plt.text(3,-7, str('case'+bestmat_str+'  '+bestrep[0:6]))
    plt.title('Best vs SFR Flux (Log-Log scale)   score = '+bestrep[0:7])
    plt.xlabel('Energy Bins (MeV)')
    plt.ylabel('Flux Data')
    plt.legend(loc='best', prop={'size': 6})
    #plt.tight_layout()
    plt.savefig('spiderbestvsSFR.png')
    plt.close()
    
    #print(len(repvals))
    #print(len(range(0,len(repvals))))
    plt.scatter(range(0,len(repvals)),repvals, s=6, label='Top 10 individuals from each generation')
    plt.title("Representativities")
    plt.legend()
    #plt.tight_layout()
    plt.savefig('spiderrepresentativities.png')
    plt.close()

Plotting()

