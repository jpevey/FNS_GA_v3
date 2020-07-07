import os
import collections
import copy
import numpy as np
import xlrd

### Setting default values

# string: Select either 6.1 or 6.2
mcnp_version = "6.1"
# Template for mcnp job scripts
### These scripts run on 1 node with 8 threads in the fill queue
### After the job ends "final result" is grepped into a file with "_done.dat"
### at the end of the input name. Used as a flag that the job is complete.
mcnp_script_templates = collections.OrderedDict()

mcnp_script_templates['6.1'] = """#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8

hostname
module unload mpi
module load intel/12.1.6
module load openmpi/1.6.5-intel-12.1
module load MCNP6

RTP="/tmp/runtp--".`date "+%R%N"`
cd $PBS_O_WORKDIR
mcnp6 TASKS 8 name=%%%INPUT%%% runtpe=$RTP
grep -a "final result" %%%INPUT%%%o > %%%INPUT%%%_done.dat
rm $RTP"""

mcnp_script_templates['6.2']= """#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8

hostname
module load MCNP6/2.0

RTP="/tmp/runtp--".`date "+%R%N"`
cd $PBS_O_WORKDIR
mcnp6 TASKS 8 name=%%%INPUT%%% runtpe=$RTP
grep -a "final result" %%%INPUT%%%o > %%%INPUT%%%_done.dat
rm $RTP"""

class mcnp_file_handler():
    def __init__(self, mcnp_version = "6.1"):
        print("Let's play with some MCNP files!")
        self.mcnp_script_template = mcnp_script_templates[mcnp_version]

    def write_mcnp_input(self, template_file, dictionary_of_replacements, input_file_str):
        print("Building MCNP model from template")

        ### Opening template file
        template_file = open(template_file, 'r')

        ### Opening output file
        output_file_string = input_file_str + ".inp"

        output_file = open(output_file_string, 'w')

        ### Moving over the template file, inserting str's based on the dictionary
        for line in template_file:
            for key in dictionary_of_replacements:
                key = str(key)
                if key in line:
                    line = line.replace(key, dictionary_of_replacements[key])
            # print("WRITING LINE TO FILE:",output_file_string, line)
            output_file.write(line)

        template_file.close()
        output_file.close()

    def build_mcnp_running_script(self, input_file_name):
        write_string = self.mcnp_script_template.replace("%%%INPUT%%%", input_file_name)
        script_file = open(input_file_name + ".sh", 'w')
        script_file.write(write_string)
        script_file.close()

    def wait_on_jobs(self, jobs_list):
        waiting_bool = True

        actively_waiting_list = copy.deepcopy(jobs_list)
        while waiting_bool:
            job_count = 0
            for file in os.listdir():
                if "_done.dat" in file:
                    for job_file in jobs_list:
                        if job_file in file:
                            job_count += 1
                            actively_waiting_list.remove(job_file)

            if job_count == len(jobs_list):
                waiting_bool = False
                print("All jobs have run, continuing")
                return
            print('Jobs complete:', job_count, 'of', len(jobs_list))
            for job_ in actively_waiting_list:
                print(job_)
            print('Waiting 15 seconds')
            os.wait(15)

    def get_flux(self, input_name):
        ### todo: write function which pulls flux tally out of mcnp output and returns it in a format required for calculate_representivity
        
        for file in os.listdir('.'):
            placeholder = file[0:4]
            ph.append(placeholder)

            if file.endswith(".inpo") == False:
                continue
        
            inputfile = open(file, 'r')
            #bins = []
            current_vals= []
            current_unc = []
            i = 0
            for line in inputfile:
                if line == '      energy   \n':
                    p = 0
                    for n in range(i+1, i+255):
                          lines = inputfile.readline(n)
                        split_l = lines.split(' ')
                        length = len(split_l)
                        #bins.append((split_l[length-5]))
                        current_vals.append(np.float64(split_l[length-2]))
                        current_unc.append(np.float64(split_l[length-1]))

                        if p == 253:
                            total.append(split_l[len(split_l)-2])
                            total_unc.append(split_l[len(split_l)-1])
                        p = p + 1
                       
                    current_vals.pop(253) #extracting last value ('total flux')
                    current_unc.pop(253)
            #bins.pop(239)
                i = i + 1
        return current_vals, current_unc
                        
        

    np.set_printoptions(precision=None, suppress=None)

    def propagate_uncertainty(self, derivative, uncertainty):
        #print(derivative)
        uncertainty = np.square(uncertainty)
        unc = np.diag(uncertainty[0])
        t_derivative = np.transpose(derivative)

        propagated_unc = np.sqrt(np.dot(derivative,np.dot(unc,t_derivative)))

        return propagated_unc
    
     def calculate_representivity(self, flux, flux_unc):
        ### show how well output is representative of sfr flux data
        
        ### SFR flux data
        workbook = xlrd.open_workbook("SFR_Flux_Data.xlsx")
        sheet = workbook.sheet_by_index(2)
        bins = []
        nums = []
        sfrflux = []
        strflux_unc = [
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

        ebins = np.array(bins)
        
        f1 = np.array(sfrflux)
        f1_unc = np.array(sfrflux_unc)
        f2 = np.array(f2)
        f2_unc = np.array(f2_unc)

        energy = ebins.reshape(1,252)
        flux1 = f1.reshape(1,252)
        flux2 = f2.reshape(1,252)
        flux1_unc = flux1 * f1_unc
        flux2_unc = flux2 * f2_unc
        #print(np.shape(flux1))
        numerator = np.dot(flux1,np.transpose(flux2))
        dnumerator_dflux1 = flux2
        dnumerator_dflux2 = flux1
        denomenator1 = np.dot(flux1,np.transpose(flux1))
        ddenomenator1_dflux1 = np.dot(2,flux1)
        ddenomenator1_dflux2 = np.zeros(np.shape(flux2))
        denomenator2 = np.dot(flux2,np.transpose(flux2))
        ddenomenator2_dflux1 = np.zeros(np.shape(flux1))
        ddenomenator2_dflux2 = np.dot(2,flux2)

        denomenator3 = np.dot(denomenator1,denomenator2)
        ddenomenator3_dflux1 = np.dot(denomenator2,(ddenomenator1_dflux1))+np.dot(denomenator1, ddenomenator2_dflux1)
        ddenomenator3_dflux2 = np.dot(denomenator2,(ddenomenator1_dflux2))+np.dot(denomenator1,ddenomenator2_dflux2)
        denomenator4 = np.sqrt(denomenator3)
        variable0 = 1/np.dot(2,np.sqrt(denomenator3))
        #variable0[np.isinf(variable0)] = 0
        ddenomenator4_flux1 = np.dot(variable0,ddenomenator3_dflux1)
        ddenomenator4_flux2 = np.dot(variable0,ddenomenator3_dflux2)
        #print(numerator, denomenator4, R)
        R = numerator/denomenator4
        dR_dflux1 = np.dot(1/(denomenator4),dnumerator_dflux1)-np.dot(numerator/denomenator4**2,ddenomenator4_flux1)
        dR_dflux2 = np.dot(1/(denomenator4),dnumerator_dflux2)-np.dot(numerator/denomenator4**2,ddenomenator4_flux2)

        uncertainty0 = np.concatenate([flux1_unc, flux2_unc])
        uncertainty = np.reshape(uncertainty0,(1,504))

        derivative0 = np.concatenate([dR_dflux1, dR_dflux2])
        derivative  = np.reshape(derivative0,(1,504))

        R_unc = propagate_uncertainty(derivative,uncertainty)   
        #print(R)
        R[np.isnan(R)] = 0
        R_unc[np.isnan(R_unc)] = 0
    return R, R_unc
        
        
    def run_mcnp_input(self, input_file):
        os.system('qsub ' + input_file + ".sh")
        print("Submitted:" , input_file + ".sh")
