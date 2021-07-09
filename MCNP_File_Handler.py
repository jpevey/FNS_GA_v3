import os
import collections
import copy
import numpy as np
import math

np.set_printoptions(precision=None, suppress=None)

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
#PBS -q corei7
#PBS -l nodes=1:ppn=8

hostname
module unload mpi
module load intel/12.1.6
module load openmpi/1.6.5-intel-12.1
module load MCNP6/1.0

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
module load MCNP6

RTP="/tmp/runtp--".`date "+%R%N"`
cd $PBS_O_WORKDIR
mcnp6 TASKS 8 name=%%%INPUT%%% runtpe=$RTP
grep -a "final result" %%%INPUT%%%o > %%%INPUT%%%_done.dat
rm $RTP"""

class mcnp_file_handler():
    def __init__(self, mcnp_version = "6.1"):
        print("Let's play with some MCNP files!")
        self.mcnp_script_template = mcnp_script_templates[mcnp_version]
        self.target_flux = 'N/A'
    def write_mcnp_input(self, template_file, dictionary_of_replacements, input_file_str):
        print("Building MCNP model from template")

        ### Opening template file
        template_file = open(template_file, 'r')

        ### Opening output file
        output_file_string = input_file_str

        output_file = open(output_file_string, 'w')

        ### Moving over the template file, inserting str's based on the dictionary
        for line in template_file:
            for key in dictionary_of_replacements:
                key = str(key)
                for split_ in line.split():
                    if key == split_:
                        line = line.replace(key, str(dictionary_of_replacements[key]))
            #print("WRITING LINE TO FILE:",output_file_string, line)
            output_file.write(line)

        template_file.close()
        output_file.close()

    def build_mcnp_running_script(self, input_file_name):
        if input_file_name.endswith('.inp') == False:
            input_file_name = input_file_name + ".inp"

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

    def get_f4_flux_from_output(self, input_name, tally_number = '4'):

        with open(input_name, 'r') as inputfile:
            in_tally = False
            in_data = False
            fluxes, uncerts = [], []

            for line in inputfile:
                line_ = line.split()

                if line.strip() == "":
                    continue

                if line_[0] == '1tally':
                    if line_[1] == tally_number:
                        if line_[2] == 'nps':
                            in_tally = True
                            continue
                if in_tally:
                    if line_[0] == 'energy':
                        in_data = True
                        continue

                    elif line_[0] == 'total':
                        print(line_, line)
                        return fluxes, uncerts, line_[1], line_[2]

                    elif in_data == True:
                        fluxes.append(line_[1])
                        uncerts.append(line_[2])

                    else:
                        continue



    def get_flux_dep(self, input_name):
        inputfile = open(input_name, 'r')

        current_vals = []
        current_unc = []
        i = 0
        for line in inputfile:
            if line.strip() == 'energy':
                p = 0
                for n in range(i, i + 253):
                    lines = inputfile.readline(n)
                    split_l = lines.split(' ')
                    length = len(split_l)
                    # bins.append((split_l[length-5]))
                    # print(split_l[length-2])
                    current_vals.append(np.float64(split_l[length - 2]))
                    current_unc.append(np.float64(split_l[length - 1]))

                    # if p == 253:
                    # total.append(split_l[len(split_l)-2])
                    # total_unc.append(split_l[len(split_l)-1])
                    p = p + 1

                total = current_vals.pop(252)  # extracting last value ('total flux')
                total_unc = current_unc.pop(252)
            # bins.pop(239)
            i = i + 1
        return current_vals, current_unc, total, total_unc


    def sum_product(self, list_1, list_2):
        sum_ = 0.0
        for item_1, item_2 in zip(list_1, list_2):
            sum_ += float(item_1) * float(item_2)
        return sum_

    def get_target_rep_values(self, target_file = "Target_Flux_Data.csv"):
        data, data_uncert = [], []
        with open(target_file) as target_flux_file:
            for line in target_flux_file:
                if line.startswith("#"):
                    continue
                line_ = line.split(',')
                try:
                    data.append(line_[1])
                    data_uncert.append(line[2])
                except:
                    pass
        return data

    def lethargy_adjustment(self, pre_adjusted_values, bins, minimum_bin_value = 1e-11):
        adjusted_values = []
        for count, val in enumerate(pre_adjusted_values):
            if count == 0:
                adjusted_values.append(float(val) / math.log10(float(bins[count]) / float(minimum_bin_value)))
            else:
                adjusted_values.append(float(val) / math.log10(float(bins[count]) / float(bins[count - 1])))
        return adjusted_values

    def calculate_representativity_v3(self, other_values, lethargy_adjustment = True, bins = []):

        ### Getting target flux value
        if self.target_flux == 'N/A':
            self.target_flux = self.get_target_rep_values()
            ### The code assumes that the targt flux passed to it is in energy, not lethargy. Makes the adjustment to lethargy.
            if lethargy_adjustment:
                assert len(bins) == len(self.target_flux), "Target flux ({}) and specified energy bins ({}) do not equal each other.".format(
                len(self.target_flux), len(bins))
                self.target_flux = self.lethargy_adjustment(self.target_flux, bins)


        if lethargy_adjustment:
            assert len(bins) > 0, "No energy bins passed during represenatitivity calculation. Needed for lethargy adjustment."
            assert len(bins) == len(other_values), "Calculated flux ({}) and specified energy bins ({}) do not equal each other.".format(
                len(other_values), len(bins))

            other_values = self.lethargy_adjustment(other_values, bins)

        val_target_to_target = self.sum_product(self.target_flux, self.target_flux)
        val_target_to_other  = self.sum_product(self.target_flux, other_values)
        val_other_to_other   = self.sum_product(other_values, other_values)
        #print(val_target_to_target, val_target_to_other, val_other_to_other)

        return val_target_to_other / math.sqrt(val_target_to_target * val_other_to_other)

    def calculate_representativity_v2(self, other_values):

        ### Getting target flux value
        if self.target_flux == 'N/A':
            self.target_flux = self.get_target_rep_values()

        val_target_to_target = self.sum_product(self.target_flux, self.target_flux)
        val_target_to_other  = self.sum_product(self.target_flux, other_values)
        val_other_to_other   = self.sum_product(other_values, other_values)
        #print(val_target_to_target, val_target_to_other, val_other_to_other)
        return val_target_to_other / math.sqrt(val_target_to_target * val_other_to_other)

    def propagate_uncertainty(self, derivative, uncertainty):
        #print(derivative)
        uncertainty = np.square(uncertainty)
        unc = np.diag(uncertainty[0])
        t_derivative = np.transpose(derivative)

        propagated_unc = np.sqrt(np.dot(derivative,np.dot(unc,t_derivative)))

        return propagated_unc


    def run_mcnp_input(self, input_file):

        if input_file.endswith('.inp') == False:
            input_file += ".inp"
        os.system('qsub ' + input_file + ".sh")
        print("Submitted:" , input_file + ".sh")

        #if location == "local":
            #os.system("mcnp6 tasks 8 inp=" + input_file)

    def get_keff(self, output_file_string):

        if output_file_string.endswith("o") == False:
            output_file_string = output_file_string + "o"

        try:
            output_file = open(output_file_string, 'r')
        except:
            print("Unable to find a keff for input " + output_file_string, "returning 10 and continuing")
            return 10.0

        found_keff = False
        for line in output_file:
            if "final result" in line:
                line_split_1 = line.split()
                keff = line_split_1[2]
                found_keff = True
        if found_keff == False:
            print("Unable to find a keff for input " + output_file_string, "returning 10 and continuing")
            return 10.0
        return keff
#
#mfh = mcnp_file_handler()

#flux, uncert, total, total_uncert = mfh.get_f4_flux_from_output("source_calc__gen_0_ind_0.inpo")

#print(mfh.calculate_representativity_v2(flux))