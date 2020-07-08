import os
import collections
import copy

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
        ###
        pass

    def calculate_representivity(self, flux):
        ### todo
        pass

    def run_mcnp_input(self, input_file):
        os.system('qsub ' + input_file + ".sh")
        print("Submitted:" , input_file + ".sh")
