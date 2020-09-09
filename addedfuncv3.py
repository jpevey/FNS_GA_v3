#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:14:01 2020

@author: CameronSalyer
"""
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import csv

def flux_plot(bins, cases_by_id, cases_to_plot, sfr_flux_plot):
    
    np.seterr(divide = 'ignore') 
    fig1, ax = plt.subplots()
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlim(10E-11, 100)
    ax.set_ylim(10E-10, 1000)
    i=0
    for case in cases_to_plot:
        parent = str(cases_by_id[i][0])
        #parent_num = get_p_num(pareto_front, parent, )
        #print(parent_num)
        
        rep = str(cases_by_id[i][2])
        tflux = str(cases_by_id[i][3])
        kval = str(cases_by_id[i][4])
        
        lab=str('Parent:'+parent+' Rep:'+rep[0:6]+' TFlux:'+tflux[0:6]+' K:'+kval[0:6])
        ax.plot(bins, case, label=lab, linewidth=2)
        i+=1
    
    
    ax.plot(bins,sfr_flux_plot, label='SFR', linewidth=2)
    ax.set_title('Best vs SFR Flux (Log-Log scale)')
    ax.set_xlabel('Energy Bins (MeV)')
    ax.set_ylabel('Flux Data')
    ax.legend(loc='upper left', prop={'size': 8})
    #ax.set_xticks(np.arange(10E-11, 100, step=10E1))
    #ax.set_yticks(np.arange(10E-10, 10, step=10E1))
    plt.tight_layout()
    fig1.savefig('GABestvsSFR.png')

def organize_bins(bins, flux):
    area = []
    new_flux = []
    i=0
    for f in flux:
        if i ==0 :
            b_low = 0
        else:
            b_low = bins[i-1]
        bin_area = f*(bins[i]-b_low)
        area.append(bin_area)
        i+=1
    total = sum(area)
    
    for f in flux:
        new_f = f/total
        new_flux.append(new_f)
    
    plot = []
    i=0
    for fl in new_flux:
        if i == 0:
            plot.append(fl/(np.log10(bins[i])))
        else:
            plot.append(fl/(np.log10(bins[i]/bins[i-1])))
        i += 1
    
    return plot
def fluxspect(bins, file):
    inputfile = open(file, 'r')
    #bins = []
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
                #bins.append(split_l[length-5])
                #print(split_l[length-2])
                current_vals.append(float(split_l[length-2]))
                current_unc.append(float(split_l[length-1]))
                #ph.append(placeholder)
                if p == 252:
                    total.append(split_l[len(split_l)-2])
                    total_unc.append(split_l[len(split_l)-1])
                p = p + 1
               
            current_vals.pop(252) #extracting last value ('total flux')
            current_unc.pop(252)
        i = i + 1

    plottable_vals = organize_bins(bins, current_vals)
    return plottable_vals

def find_best(data):
    #Pull SFR data to compare for flux spect
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
    sfr_flux_plot = organize_bins(bins, sfrflux)
    
    cases = []
    rep = []
    keff = []
    totflux = []
    ave_score = []
    placement = []
    for d in data:
        average = (float(d[2])+float(d[3])+float(d[4]))/3
        placement.append(d[0])
        cases.append(d[1])
        rep.append(float(d[2]))
        keff.append(float(d[3]))
        totflux.append(float(d[4]))
        ave_score.append(average)
        
    max_rep = max(rep)
    #max_keff = max(keff)
    max_totflux = max(totflux)
    max_ave = max(ave_score)
    #min_ave = min(ave_score)
    for i in range(0,len(data)):
        if rep[i] == max_rep:
            high_rep = fluxspect(bins,cases[i])
            highest_rep_id = placement[i], cases[i],rep[i],keff[i],totflux[i]
        #if keff[i] == max_keff:
            #high_keff = fluxspect(bins,cases[i])
            #highest_keff_id = i, cases[i],rep[i],keff[i],totflux[i]
        if totflux[i] == max_totflux:
            high_flux = fluxspect(bins,cases[i])
            highest_flux_id = placement[i], cases[i],rep[i],keff[i],totflux[i]
        if ave_score[i] == max_ave:
            high_ave = fluxspect(bins,cases[i])
            highest_ave_id = placement[i], cases[i],rep[i],keff[i],totflux[i]
        #if ave_score[i] == min_ave:
            #low_ave = fluxspect(bins,cases[i])
            #lowest_ave_id = i, cases[i],rep[i],keff[i],totflux[i]
        
        
    cases_to_plot = high_rep, high_flux, high_ave#, low_ave, high_keff
    cases_by_id = highest_rep_id, highest_flux_id, highest_ave_id#, lowest_ave_id, highest_keff_id
    
    flux_plot(bins, cases_by_id, cases_to_plot, sfr_flux_plot)
    return cases_to_plot, cases_by_id
def material_count(matlist):
    v = 0
    p = 0
    f = 0
    s = 0
    for m in matlist:
        m = int(m)
        if m == 1:
            v+=1
        if m == 2:
            p+=1
        if m == 3:
            f+=1
        if m == 4:
            s+=1
    return v, p, f, s

def pre_plotting(all_data, data):
    #finding last gen automatically
    lastgen = max(all_data[6])
    #print(lastgen)
    pareto_front = []
    x_vals = []
    y_vals = []
    z_vals = []
    materials = []
    p=0
    for i in range(0, len(data)):
        #for dat in data[0]:
        #print(data[i][0])
        if data[i][0] == str(lastgen):##add automatic finding of last generation
            #print(i)
            best = p, data[i][1], float(data[i][2]), float(data[i][3]), float(data[i][5]), int(data[i][4])
            x_vals.append(float(data[i][2])) #rep
            y_vals.append(float(data[i][3])) #total flux
            z_vals.append(float(data[i][5])) #keff
            #print(mat_used[i])
            materials.append(all_data[5][i])
            pareto_front.append(best)
            p+=1

    #print(type(pareto_front[0][1]))
    plt.scatter(all_data[1], all_data[2], color='b', s=8, label="All Parents")
    plt.scatter(x_vals, y_vals, color='r', s=10, label="Final Generation Parents")
    plt.title("Representativity vs Total Flux")
    plt.xlim(.3,1)
    plt.ylim(0, .037)
    plt.xticks(np.arange(.3,1.1, step=0.1))
    plt.yticks(np.arange(.0, .037, step=0.007))
    plt.xlabel('Representativity')
    plt.ylabel('Total flux')
    plt.legend(loc='upper left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig('pareto_rep_vs_totalflux_fianlgen.png')
    plt.close()
    #print(z_vals)
    # Rearranging parents for better representation
    #print(x_vals)
    y_vals, x_vals, z_vals, materials, pareto_front = zip(*sorted(zip(y_vals, x_vals, z_vals, materials,pareto_front), reverse=True))
    #print(x_vals)
    #Plotting keff, total flux, and rep on same plot
    ind = np.arange(0,len(pareto_front))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(.6, 1)
    ax2.set_ylim(0, .035)
    ln1 = ax1.plot(ind, x_vals, color='r', label='Representativity') #representativity
    ln2 = ax1.plot(ind, z_vals, color='b',label='Keff')
    ln3 = ax2.plot(ind, y_vals, color='g',label='Total Flux') #total flux
    
    #combinging legends for double y-axis plot
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower center')
    
    ax1.set_xticks(np.arange(0,len(ind), step=1))
    ax1.set_xlabel('Parent #')
    ax1.set_ylabel('Rep, Keff')
    ax2.set_ylabel('Total Flux')
    ax1.set_title('K, Fast Flux, and Representativity for Final Gen. Parents')
    plt.tight_layout()
    fig.savefig('k_ff_rep_finalgenparents.png')
    plt.close()
    
    #Materials Plot (Plate count for gen 50)
    
    num_void = []
    num_poly = []
    num_fuel = []
    num_sodium = []
    for i in ind:
        #print(materials[i-1])
        void, poly, fuel, sodium = material_count(materials[i])
        num_void.append(void)
        num_poly.append(poly)
        num_fuel.append(fuel)
        num_sodium.append(sodium)
        
        
    plt.plot(ind, num_void, label='Void')
    plt.plot(ind, num_poly, label='Poly')
    plt.plot(ind, num_fuel, label='Fuel')
    plt.plot(ind, num_sodium, label='Sodium')
    plt.title("Disk Counts for Final Gen. Parents")
    plt.xlim(0,len(ind))
    plt.xticks(np.arange(0, len(ind), step=1))
    plt.yticks(np.arange(0, 20, step=2))
    plt.xlabel('Parent #')
    plt.ylabel('Number of Disks')
    plt.legend(loc='center right', prop={'size': 8})
    plt.tight_layout()
    plt.savefig('DiskCount_Gen50.png')
    plt.close()
    return pareto_front

def pull_data():
    
    #print(total_sfr)
    k_eff = []
    gen_num = []
    repvals = []
    cases = []
    total_flux = []
    front_val = []
    mat_used = []
    with open('_output.csv') as csv_file:
        start=100 #random starter
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:               
            #print(row)
            #if row[0] == 'fuel_index_multiplier':
                #print(line_count)
            #print(row)
            if row == ['use_crowding_distance', 'True']:
                start = line_count
            if line_count > start:
                #print(row)
                if row[4] == any(row):
                    continue
                if row[4] == 'N/A':
                    continue
                #if float(row[3]) > .95:
                    #keff.append(float(row[3]))
                else:
                    gen = int(row[0])
                    k = float(row[3])
                    rep = float(row[4])
                    total = float(row[5])
                    frontval = int(row[6])
                    mat = row[11:len(row)-1]
                total_flux.append(total)
                k_eff.append(k)
                repvals.append(rep)
                front_val.append(frontval)
                gen_num.append(gen)
                mat_used.append(mat)
                cases.append(str(row[2]+'o')) 
            line_count += 1
    #print(len(repvals), len(keff))
    i=0
    for case in cases:
        if repvals[i] == 0 and total_flux[i] == 0:
            cases.pop(i)
            repvals.pop(i)
            total_flux.pop(i)
            front_val.pop(i)
            gen_num.pop(i)
            k_eff.pop(i)
            mat_used.pop(i)

        i+=1
    all_data = cases, repvals, total_flux, front_val, k_eff, mat_used, gen_num
    #print(mat_used)
    rep_flux = np.array([gen_num, cases, repvals, total_flux, front_val, k_eff])##change keff to total flux
    data = np.transpose(rep_flux)
    
    return all_data, data

def run_postproc():
    
    all_data, data = pull_data()
    pareto_front = pre_plotting(all_data, data)
    find_best(pareto_front)
    return pareto_front
    
#pareto = run_postproc()
    
    
