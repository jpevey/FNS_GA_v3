#!/bin/bash
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
mcnp6 TASKS 8 name=stacked_cylinders_test__gen_1_ind_5.inp runtpe=$RTP
grep -a "final result" stacked_cylinders_test__gen_1_ind_5.inpo > stacked_cylinders_test__gen_1_ind_5.inp_done.dat
rm $RTP