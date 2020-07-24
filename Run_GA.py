import collections
import Genetic_Algorithm
import Individual_v1

options = collections.OrderedDict()
options['skip_waiting_on_jobs_debug'] = True
options['fake_fitness_debug'] = True
options['skip_writing_files'] = False
options['verify_fuel_mass_after_mutation'] = False
options['verify_fuel_mass_after_crossover'] = False
options['enforce_fuel_count'] = False
options['enforce_material_number'] = 1
options['enforced_fuel_count_value'] = 0
options['include_pattern'] = False
options['number_of_generations'] = 1
options['number_of_individuals'] = 3
options['number_of_parents'] = 2
options['minimum_fuel_elements'] = 0
options['maximum_fuel_elements'] = 3
options['remake_duplicate_children'] = True
options['mutation_rate'] = 0.05 # for each individual, % chance that a material flips, 0.05 = 5%
options['mutation_type'] = 'bitwise'  # bitwise - each material has a chance
                                      # to mutate to other material based on mutation_rate
options['material_types'] = [1, 2, 3, 4]
# When doing crossover, select a parent based on specified method (default is uniformly at random)
# Parent_2 is choosen at random with probabilities based on the diversity score
# For example, given 3 parents the first is choosen at random (default)
# Then, starting with the most dissimilar parent, a chance of choosing each parent based on
# (diversity score / sum of all diversity scores)
options['choose_parent_based_on_bitwise_diversity'] = True
options['crossover_type'] = 'bitwise'  # bitwise - each material bit has a chance to come from parent 1 or parent 2
# options['crossover_type'] = 'singlepoint' # bitwise - each material bit has a chance to come from parent 1 or parent 2

options['grid_x'] = 30  # specifies how big a grid to create for this geometry
options['grid_y'] = 1
# options['grid_z'] = 1 need to code 3rd dimension for fns
options['total_materials'] = options['grid_x'] * options['grid_y']
options['scale_template_file_string'] = '11x11_grid_array_template.inp'
options['mcnp_template_file_string'] = 'simplecyl_30x15in.inp'
options['mcnp_keff_template_file_string'] = 'simplecyl_keff_30x15in.inp'
options['file_keyword'] = 'stacked_cylinders_test_'
options['solver'] = 'mcnp'
# solver_location: 'local' or 'necluster'
options['store_all_individuals'] = False
options['solver_location'] = 'necluster'
# options['solver'] = 'cnn'
# options['geometry'] = 'cyl'
options['geometry'] = 'grid'
options['template_keywords'] = collections.OrderedDict()
### Creating keywords list for template file
options['keywords_list'] = []
options['keywords_template'] = 'mat_'
for x in range(options['grid_x']):
    options['keywords_list'].append(options['keywords_template'] + str(x+1))
for val in options['keywords_list']:
    options['template_keywords'][val] = ""
options['write_output_csv'] = True
options['output_filename'] = '_output'
### Currently uses single fitness function, 'keff' (doesn't work yet), or 'representativity'
options['fitness'] = ['keff#threshold','representativity']
options['fitness_sort_by'] = 'representativity'
options['default_mcnp_mat_count_and_density'] = collections.OrderedDict()
options['default_mcnp_mat_count_and_density'][1] = '0'
options['default_mcnp_mat_count_and_density'][2] = '2 -0.93'
options['default_mcnp_mat_count_and_density'][3] = '3 -18.94'
options['default_mcnp_mat_count_and_density'][4] = '4 -0.910'
options['output_writeout_values'] = ['generation', 'individual_count', 'input_name', 'representativity', 'number_of_fuel',
                                     'write_out_parents', 'write_out_average_diversity_score', 'materials']
# options['scale_script_template'] = \
# """#!/bin/bash
#
##PBS -q corei7
##PBS -V
##PBS -l nodes=1:ppn=1
#
# module load scale/6.2.3
#
# cd $PBS_O_WORKDIR
#
# scalerte -m %%%input%%%
# grep -a "final result" %%%input%%%.out > %%%input%%%_done.dat"""
options['scale_script_template'] = \
    """
    #!/bin/bash
    #PBS -q fill
    #PBS -V
    #PBS -l nodes=1:ppn=8
    
    module unload mpi
    module load openmpi/1.8.8-gnu
    module load scale/dev
    cat ${PBS_NODEFILE}
    #NP=$(grep -c node ${PBS_NODEFILE})
    
    cd $PBS_O_WORKDIR
    
    #echo $NP
    scalerte -m -N 8 -M ${PBS_NODEFILE} -T /home/tmp_scale/$USER/scale.$$ %%%input%%%
    grep -a "final result" %%%input%%%.out > %%%input%%%_done.dat
    """
options['pattern_to_include'] = \
    [[[1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1],
      [2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1],
      [1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2],
      [1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1],
      [2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1],
      [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2],
      [2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1],
      [1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2],
      [2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2],
      [2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2],
      [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]]
options['read_parents_from_file'] = False
if options['read_parents_from_file']:
    read_in_file = open('readin_parents.csv')
    options['pattern_to_include'] = []
    for line in read_in_file:
        print(line)
        line_split = line.split(',')
        whole_pattern = []
        row_pattern = []
        count = 0
        for val in line_split:
            row_pattern.append(int(val))
            count += 1
            if count == options['grid_x']:
                count = 0
                whole_pattern.append(row_pattern)
                row_pattern = []

        options['pattern_to_include'].append(whole_pattern)

options['cyl_scale_template'] = [
    'cylinder material 1 10.83396326 2.54 0.0',
    'cylinder material 1 15.32153778 2.54 0.0',
    'cylinder material 1 18.76497482 2.54 0.0',
    'cylinder material 1 21.66792653 2.54 0.0',
    'cylinder material 1 24.22547832 2.54 0.0',
    'cylinder material 1 26.53768189 2.54 0.0',
    'cylinder material 1 28.66397251 2.54 0.0',
    'cylinder material 1 30.64307556 2.54 0.0',
    'cylinder material 1 32.50188979 2.54 0.0',
    'cylinder material 1 34.26 2.54 0.0',
    'cylinder material 1 35.93219114 2.54 0.0',
    'cylinder material 1 37.52994964 2.54 0.0',
    'cylinder material 1 39.06241006 2.54 0.0',
    'cylinder material 1 40.53697867 2.54 0.0',
    'cylinder material 1 41.95975929 2.54 0.0',
    'cylinder material 1 43.33585305 2.54 0.0',
    'cylinder material 1 44.66957488 2.54 0.0',
    'cylinder material 1 45.96461335 2.54 0.0',
    'cylinder material 1 47.22415102 2.54 0.0',
    'cylinder material 1 48.45095663 2.54 0.0'
]

options['check_eigenvalue'] = True
options['check_eigenvalue_function'] = 'enforced_maximum_eigenvalue'
options['enforced_maximum_eigenvalue'] = 0.98
options['fuel_index'] = 3
options['fuel_index_multiplier'] = 2.54 / 2

if __name__ == '__main__':
    ga = Genetic_Algorithm.genetic_algorithm(options)