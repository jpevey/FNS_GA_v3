import os
import collections
import math
import random
import copy
import time


import Individual_v1 as individual
import MCNP_File_Handler
#import CNN_Handler

class genetic_algorithm:
    def __init__(self, options_dict):
        self.mcnp_file_handler = MCNP_File_Handler.mcnp_file_handler()
        print("Initializing GA with:", options_dict)
        self.options = options_dict
        ### List of current generation individuals
        self.individuals = []
        self.mcnp_inputs = []

        random.seed(self.options['python_random_number_seed'])

        self.generation = 0
        self.individual_count = 0
        ### Creating initial population
        for ind in range(self.options['number_of_individuals']):
            ind_ = individual.individual(options_dict, self.generation, self.individual_count)
            self.individuals.append(ind_)
            print("outside ind matrix", ind_.material_matrix)

            self.individual_count += 1

        for ind in self.individuals:
            print(ind.material_matrix)

        ### Loading CNN if needed
        #if 'cnn' in self.options['solver']:
        #    model_string = "CNN_3d_11x11_fm_cad_4x4_kern_v2.hdf5"
        #    self.cnn_handler = CNN_Handler.CNN_handler(model_string)
        #    self.cnn_input = []

        if self.options['include_pattern']:
            print("Including a pattern in initial population!")
            for ind_count, pattern_to_include in enumerate(self.options['pattern_to_include']):
                self.individuals[ind_count].material_matrix = []
                for material in pattern_to_include:
                    self.individuals[ind_count].material_matrix.append([material])
                self.individuals[ind_count].make_material_string_scale('%array%1')

        ### Creating output csv if needed
        if self.options['write_output_csv']:
            output_csv = open(self.options['output_filename'] + '.csv', 'w')
            ### Writing out options for this run
            for flag in self.options:
                output_csv.write("{},{}\n".format(flag, self.options[flag]))
            output_csv.write("%%%begin_data%%%\n")
            output_csv.write(self.create_header() +"\n")
            output_csv.close()


        ### Running eigenvalue calcs if needed
        #if self.options['enforced_maximum_eigenvalue'] == True:
        #    getattr(self, self.options['check_eigenvalue_function'])()

        ### Evaluating initial population, gen 0
        print("Evaluating initial population")
        self.evaluate(self.options['fitness'])

        if self.options['remake_duplicate_children'] == True:
            self.all_individuals = copy.deepcopy(self.individuals)
            print("All individuals:", self.all_individuals)

        if self.options['use_non_dominated_sorting'] == True:
            self.parents_list = self.non_dominated_sorting()
        else:
            self.individuals.sort(key=lambda x: getattr(x, self.options['fitness_sort_by']), reverse=True)
            ### Pairing down individuals to be specified number
            self.individuals = self.individuals[:self.options['number_of_individuals']]


        ### Evaluating diversity of population
        if self.options['choose_parent_based_on_bitwise_diversity']:
            print("Evaluating diversity of parents")
            self.evaluate_bitwise_diversity_of_parents()

        #self.write_output_v2(self.individuals)
        self.generation += 1

        ### Running GA algo
        for generation in range(self.options['number_of_generations']):
            print("Generation: ", self.generation)
            for constraint in self.options['constraint']:
                if "linearSimAnneal" in constraint:
                    print("Applying linearSimAnneal to all previous individuals")
                    self.all_individuals = self.apply_constraint(self.all_individuals, constraint)

            print("crossover")
            if self.options['adjustable_zone_2A_cassette_bool']:
                list_of_children = self.crossover()
            else:
                list_of_children = self.crossover()
            print("mutating")
            list_of_mutated_children = self.mutate(list_of_children, 'material_matrix')

            if self.options['adjustable_zone_2A_cassette_bool']:
                list_of_mutated_children = self.mutate(list_of_mutated_children,
                                                       'material_matrix_cassette_2A',
                                                       do_variable_size_mutation = True)

            if self.options['remake_duplicate_children']:
                list_of_mutated_children = self.remake_duplicate_children(list_of_mutated_children,
                                                                          self.all_individuals,
                                                                          debug = False)

            if self.options['enforce_material_count_before_evaluation']:
                print("enforcing fuel count:", self.options['enforced_fuel_count_value'])
                for ind_count, ind in enumerate(list_of_mutated_children):
                    ind.enforce_material_count(self.options['enforce_material_number'], self.options['enforced_fuel_count_value'])




            print("evaluating children")
            evaluated_children = self.evaluate(self.options['fitness'], list_of_mutated_children)

            ### Adding now evaluated children to master list of all individuals
            self.all_individuals += evaluated_children


            ### combining now evaluated children with previous list of parents
            self.individuals = self.parents_list + evaluated_children





            #for ind_count, ind_ in enumerate(self.individuals):
                #print(ind_count, ind_.ind_count, ind_.generation, ind_.keff, ind_.representativity)

            print("Write output")
            if self.options['write_output_csv']:
                if self.options['sort_all_possible_individuals']:
                    self.write_output_v2(self.parents_list+evaluated_children)
                else:
                    self.write_output_v2(self.individuals)

            self.generation += 1

            #print("Printing all individuals!")
            #for ind_ in self.all_individuals:
                #print(ind_.input_file_string, ind_.representativity)

        if self.options['output_all_individuals_at_end_of_calculation'] == True:
            output_csv = open(self.options['output_all_individuals_at_end_of_calculation_file_name'] + '.csv', 'w')
            for flag in self.options:
                output_csv.write("{},{}\n".format(flag, self.options[flag]))
            output_csv.close()
            print(self.options['output_all_individuals_at_end_of_calculation_file_name'])
            self.write_output_v2(list_of_individuals = self.all_individuals, output_file_name = self.options['output_all_individuals_at_end_of_calculation_file_name'])


    def apply_constraint(self, list_of_individuals, constraint_type):
        print("Applying constraint:", constraint_type)
        meet_constraint = False
        ### Seperating constraint and options, etc.
        constraint_split = constraint_type.split("#")

        constraint_type_ = constraint_split[0]
        constraint_run_location = constraint_split[1]
        constraint_options = constraint_split[2]

        if constraint_type_ == 'keff':
            if 'mcnp' in self.options['solver']:
                self.mcnp_keff_inputs = []
                self.mcnp_file_handler = MCNP_File_Handler.mcnp_file_handler()
                for individual in list_of_individuals:
                    ### Building MCNP input file
                    ### Building initial dict of what to replace in template input file
                    data_dictionary_ = individual.create_discrete_material_mcnp_dictionary(
                        material_matrix_val = 'material_matrix',
                        keywords_list = self.options['keywords_list'])

                    ### If doing a variable cassette 2A, adding those values to the data_dictionary_
                    if self.options['adjustable_zone_2A_cassette_bool']:
                        data_dictionary_ = individual.create_discrete_material_mcnp_dictionary(
                            material_matrix_val = 'material_matrix_cassette_2A',
                            keywords_list = self.options['adjustable_zone_2A_keywords_list'],
                            data_dict = data_dictionary_)


                        data_dictionary_ = individual.build_variable_cassette_2a_dictionary(data_dictionary_)


                    ### Finding and adding the fuel location to geometry dictionary
                    data_dictionary_['kcode_source_x'] = str(individual.find_fuel_location())



                    ### Building MCNP input
                    self.mcnp_file_handler.write_mcnp_input(
                        template_file=self.options['mcnp_keff_template_file_string'],
                        dictionary_of_replacements=data_dictionary_,
                        input_file_str=individual.keff_input_file_string)
                    ### Building MCNP input script for cluster
                    self.mcnp_file_handler.build_mcnp_running_script(individual.keff_input_file_string)
                    ### Running MCNP input
                    self.mcnp_file_handler.run_mcnp_input(individual.keff_input_file_string)
                    ### Adding input name to list, used to determine if the jobs have completed or not
                    self.mcnp_keff_inputs.append(individual.keff_input_file_string)

                ### Waits on the jobs to be completed (looking for "_done.dat" files)
                self.wait_on_jobs('mcnp_keff')

                ### Grabs keff from the output files
                for ind in list_of_individuals:
                    if self.options['fake_fitness_debug']:
                        ind.keff = random.uniform(0.1, 1.0)
                    else:
                        ind.keff = self.mcnp_file_handler.get_keff(ind.keff_input_file_string)

                    ### If that keff is above a set threshold, sets acceptable_eigenvalue to false. Else, sets it True.
                    if float(ind.keff) >= self.options['enforced_maximum_eigenvalue']:
                        print("keff, ", ind.keff, "too high. Skipping source calculation")
                        ind.acceptable_eigenvalue = False
                    else:
                        ind.acceptable_eigenvalue = True
            #if 'scale' in self.options['solver']:
                ### create scale inputs, add filenames to list
                #for individual in self.individuals:
                    #if individual.evaluated_keff == False:
                        #if self.options['geometry'] == 'cyl':
                        #    individual.make_material_string_scale('cyl_materials')
                        #elif self.options['geometry'] == 'grid':
                        #    individual.make_material_string_scale('%array%1')
                        #else:
                        #    print("Geometry not handled in evaluate function")
                        #    exit()
                        #scale_inputs.append(individual.setup_scale(self.generation))
                        #individual.evaluated_keff = True
                        #if self.options['fake_fitness_debug']:
                        #    individual.keff = random.uniform(0.5, 1.5)
                #self.scale_inputs = scale_inputs
                ### submitting all jobs and waiting on all jobs
                #if self.options['solver_location'] == 'necluster':
                #    self.submit_jobs(self.scale_inputs)
                #    self.wait_on_jobs('scale')

                #if self.options['solver_location'] == 'local':
                #    print("Cant run scale locally... yet... fix this")
                #    exit()

                #for individual in self.individuals:
                #    individual.get_scale_keff()

        if constraint_type_ == 'representativity':
            constraint_options_split = constraint_options.split(",")

            ### This linear constraint applies a penalty to individuals whose
            ### objective function is not at a threshold.
            ### With linearSimAnneal this threshold builds linearly to be in full
            ### effect at generation specified in the second option "gen_XXX"
            ### The target is given in the second option as
            ### "(target objective function)_(target value)"
            ### Example: "representativity#postevaluate#linearSimAnneal,gen_50,val_0.95"

            if constraint_options_split[0] == "linearSimAnneal":
                print("Applying linearSimAnneal constraint")
                assert len(constraint_options_split) == 3, "Not enough options specified in linearSimAnneal"

                _, gen_val = constraint_options_split[1].split('_')
                _, target_val = constraint_options_split[2].split('_')

                gen_val = int(gen_val)
                target_val = float(target_val)

                ### Setting linearSimAnneal_threshold based on current generation
                ### If beyond the specified generation, setting it to be the target_val
                try:
                    linearSimAnneal_threshold = (self.generation / gen_val) * target_val
                except ZeroDivisionError:
                    linearSimAnneal_threshold = 0.0
                if self.generation >= gen_val:
                    linearSimAnneal_threshold = target_val

                ### Applying linearSimAnneal constraint to individuals
                for individual in list_of_individuals:
                    if getattr(individual, constraint_type_) < linearSimAnneal_threshold:
                        for objective_function in self.options['fitness']:
                            setattr(individual, objective_function, 0.0)

                        print("Setting individual:", individual.ind_count, getattr(individual, constraint_type_),constraint_type_,'to 0.0', linearSimAnneal_threshold)
                    else:
                        print("Individual Ok!", individual.input_file_string, getattr(individual, constraint_type_), 'greater than', linearSimAnneal_threshold)

        return list_of_individuals

    def non_dominated_sorting(self):
        front = [[]]

        individuals_to_sort = self.individuals
        if self.options['sort_all_possible_individuals']:
            individuals_to_sort = self.all_individuals

        for individual in individuals_to_sort:
            individual.front_rank = 'none'
            print(individual.ind_count)
            print("Nondominated sorting", individual.ind_count, individual.keff, individual.representativity)
            ### Initializing lists
            dominated_list = []
            number_of_inds_that_dominate_individual = 0
            #for fitness_ in self.options['fitness']:
            #dominated_list.append([])
            #number_of_inds_that_dominate_individual.append(0)

            ### Iterating over all individuals, comparing fitnesses
            for individual_ in individuals_to_sort:
                if individual == individual_:
                    continue
                individual_.front_number = 0

                if self.check_domination(individual, individual_):
                    if individual_ not in dominated_list:
                        dominated_list.append(individual_)

                elif self.check_domination(individual_, individual):
                    number_of_inds_that_dominate_individual += 1


            individual.dominated_list = dominated_list
            individual.number_of_inds_that_dominate_individual = number_of_inds_that_dominate_individual
            print(individual.input_file_string, individual.keff, individual.representativity, len(individual.dominated_list), individual.number_of_inds_that_dominate_individual)

            #for fitness_index, fitness_ in enumerate(self.options['fitness']):
            if individual.number_of_inds_that_dominate_individual == 0:
                individual.front_rank = 0
                #print("Ind is rank one:", individual.input_file_string, individual.representativity, individual.keff)
                front[0].append(individual)
        ### Front counter
        pareto_front = 0
        while front[pareto_front] != []:
            current_front = []
            for individual in front[pareto_front]:
                #print(individual.input_file_string, "dominates:")
                for individual_ in individual.dominated_list:
                    if individual_.input_file_string == individual.input_file_string:
                        continue
                    #print(individual_.input_file_string, individual_.number_of_inds_that_dominate_individual)
                    individual_.number_of_inds_that_dominate_individual -= 1
                    #print(individual_.input_file_string, individual_.number_of_inds_that_dominate_individual)
                    if individual_.number_of_inds_that_dominate_individual == 0:

                        if individual_.front_rank == 'none':
                            individual_.front_rank = pareto_front + 1
                            current_front.append(individual_)
                    #print("Ranks:", individual_.input_file_string, individual_.front_rank, current_front)

            front.append(current_front)
            pareto_front += 1
        print("Pareto fronts:")
        for front_count, front_list in enumerate(front):
            print(front_list)
            for ind in front_list:
                print(front_count, ind.input_file_string, ind.representativity, ind.keff)
        self.pareto_front = front

        #for individual_ in self.individuals:
            #print(individual_.input_file_string,  individual_.representativity, individual_.keff)


        ### Building list of parents
        parents_list = []
        for front in self.pareto_front:
            if front == []:
                continue
            #print("Front:", len(front))
            if len(front) < (self.options['number_of_parents'] - len(parents_list)):
                #print("Length of parent list!!!", len(parents_list))
                parents_list = parents_list + front
                #print("Length of parent list!!!", len(parents_list))
            else:

                front = self.crowding_distance(front)
                #print(len(front), self.options['number_of_parents'], len(parents_list))
                front.sort(key=lambda x: x.crowding_distance, reverse=True)

                ind_count = 0
                #print("Adding parents to parents list")
                while self.options['number_of_parents'] != len(parents_list):
                    parents_list.append(front[ind_count])
                    ind_count += 1
            #print("parent_list len, etc", len(parents_list), self.options['number_of_parents'])
        #for parent in parents_list:
            #print(parent.input_file_string, parent.keff, parent.representativity, parent.crowding_distance)
        return parents_list

    def crowding_distance(self, front):
        if front == []:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for fitness in self.options['fitness']:
            #print("fitness",fitness)
            if len(front) == 0:
                continue
            if "#" in fitness:
                fitness = fitness.split("#")
                fitness = fitness[0]

            ### Setting highest and lowest ind's to have large crowding distance
            front.sort(key=lambda x: getattr(x, fitness), reverse=True)
            front[0].crowding_distance = 99999999999999999.0
            front[-1].crowding_distance = 99999999999999999.0
            max_value = float(getattr(front[0],fitness))
            min_value = float(getattr(front[-1], fitness))
            diff_max_min = max_value - min_value

            ### Adding crowding distance for this fitness
            for count, ind in enumerate(front):

                if count == 0:
                    continue
                if count == (len(front) - 1):
                    continue

                ind_n_plus_one  = float(getattr(front[count-1],fitness))
                ind_n_minus_one = float(getattr(front[count+1],fitness))
                try:
                    ind.crowding_distance = ind.crowding_distance + (ind_n_plus_one - ind_n_minus_one)/(diff_max_min)
                    if ind.crowding_distance < 0.0:
                        print("Crowding distance below 0!", ind.crowding_distance)
                        exit()
                except:
                    continue
                print("CROWDING DISTANCE!!!", ind.crowding_distance, count, len(front))

        return front

### Checks if ind_1 dominates ind_2
    def check_domination(self, ind_1, ind_2):
        for fit_count, fitness_ in enumerate(self.options['fitness']):
            if "#" in fitness_:
                fitness_ = fitness_.split("#")
                fitness_ = fitness_[0]

            ind_value = getattr(ind_1, fitness_)
            comparison_value = getattr(ind_2, fitness_)

            if float(ind_value) >= float(comparison_value):
                continue
            else:
                return False

        return True

    def remake_duplicate_children(self, list_of_children, comparison_list, debug = True):
        if debug:
            print("".format())
        for child in list_of_children:
            if debug:
                print("Child: {}".format(child.ind_count))
            for comparison_ind in comparison_list:
                if debug:
                    print("Comparing to ind:{}".format(comparison_ind.ind_count))
                ### Setting comparison score to 0
                comparison_score = 0

                ### Setting maximum score (if all plates are exactly the same)
                maximum_score = self.options['total_materials']

                ### If doing a variable cassette 2A, adding that length to the maximum score
                if self.options['adjustable_zone_2A_cassette_bool']:
                    maximum_score += len(child.material_matrix_cassette_2A)

                if debug:
                    print("Maximum possible score:{}".format(maximum_score))
                for child_mat, comp_mat in zip(child.material_matrix, comparison_ind.material_matrix):
                    if child_mat == comp_mat:
                        comparison_score += 1
                if debug:
                    print("    Score after material matrix check:{}".format(comparison_score))
                if self.options['adjustable_zone_2A_cassette_bool']:
                    for child_mat, comp_mat in zip(child.material_matrix_cassette_2A, comparison_ind.material_matrix_cassette_2A):
                        if child_mat == comp_mat:
                            comparison_score += 1
                    if debug:
                        print("    Score after variable cassette 2A check:{}".format(comparison_score))
                if debug:
                    print("Score: {} Max Score: {}".format(comparison_score, maximum_score))
                if comparison_score == maximum_score:

                    child.material_matrix = self.single_bit_mutation(child.material_matrix, force_mutation=True, force_mutation_per_material_sublist=2)

        return list_of_children

    def evaluate_bitwise_diversity_of_parents(self):

        temp_material_master_list = []
        for individual_count, individual in enumerate(self.individuals):
            if individual_count > self.options['number_of_parents'] - 1:
                individual.total_diversity_score = "N/A"
                continue

            individual.bitwise_diversity_scores = []
            ### cycling over parents evaluating the diversity, if a score of [total number of material locations] is given
            ### if the individuals are exactly the same, 0 exactly opposite

            for comparison_count, comparison_individual in enumerate(self.individuals):
                if comparison_count == individual_count:
                    continue
                if comparison_count > self.options['number_of_parents'] - 1:
                    continue
                individual.bitwise_diversity_scores.append([comparison_count,
                                                            self.evaluate_bitwise_diversity(individual,
                                                                                            comparison_individual)])

            ### Calculating total diversity score
            # print("Calculating total diversity score")
            total_diversity_score = 0
            for d_s in individual.bitwise_diversity_scores:
                # print(d_s[1])
                total_diversity_score += d_s[1]

            individual.total_diversity_score = total_diversity_score
            # print(individual.total_diversity_score)

    def evaluate_bitwise_diversity(self, individual, comparison_individual):
        score = self.options['total_materials']
        for material_list_count, material_list in enumerate(individual.material_matrix):
            for material_count, material in enumerate(material_list):
                individual_material = individual.material_matrix[material_list_count][material_count]
                comparison_material = comparison_individual.material_matrix[material_list_count][material_count]

                if individual_material == comparison_material:
                    score -= 1
        return score

    def choose_parent_based_on_bitwise_diversity(self, parent_1_value):
        diversity_scores = self.individuals[parent_1_value].bitwise_diversity_scores

        ### Sorting diversity scores
        diversity_scores.sort(key=lambda x: x[1], reverse=True)

        ### Setting the parent_2 default value to the most similar case
        parent_2_index = diversity_scores[self.options['number_of_parents'] - 2][0]

        for d_s in diversity_scores:
            random_value = random.uniform(0, 1)

            try:
                if random_value < (d_s[1] / self.individuals[parent_1_value].total_diversity_score):
                    parent_2_index = d_s[0]
            except:
                parent_2_index = d_s[0]
                return parent_2_index


        return parent_2_index

    def evaluate(self, evaluation_types, list_of_individuals = "Default"):
        if list_of_individuals == "Default":
            list_of_individuals = self.individuals



        ### Updating list of individuals based on whether they meet the constraints in the constraint options
        for constraint in self.options['constraint']:
            if 'preevaluate' in constraint:
                list_of_individuals = self.apply_constraint(list_of_individuals, constraint)

        ### This loop builds and runs required MCNP files to evaluate individuals

        for evaluation_type in evaluation_types:
            ### Getting options from 'fitness' list
            if "#" in evaluation_type:
                evaluation_type, evaluation_options = evaluation_type.split("#")


            if evaluation_type == 'representativity':
                print("Evaluating Representativity")
                if 'mcnp' in self.options['solver']:
                    self.mcnp_inputs = []
                    for individual in list_of_individuals:
                        if individual.ran_source_calculation == False:
                            if individual.acceptable_eigenvalue == True:
                                ### Building and submitting MCNP input file
                                self.build_and_run_mcnp_evaluation_input(individual)
                                individual.ran_source_calculations = True
                    self.wait_on_jobs('mcnp')

            if evaluation_type == 'total_flux':
                print("Evaluating Total Flux")
                self.mcnp_inputs = []
                if 'mcnp' in self.options['solver']:
                    for individual in list_of_individuals:
                        if individual.ran_source_calculation == False:
                            if individual.acceptable_eigenvalue == True:
                                ### Building and submitting MCNP input file
                                self.build_and_run_mcnp_evaluation_input(individual)
                                individual.ran_source_calculations = True
                    self.wait_on_jobs('mcnp')

            if evaluation_type == 'integral_keff':
                print("Evaluating integral keff of individuals")
                self.mcnp_inputs = []
                if 'mcnp' in self.options['solver']:
                    for individual in list_of_individuals:
                        if individual.acceptable_eigenvalue == True:
                            ### Building and submitting MCNP input file
                            self.build_and_run_mcnp_evaluation_input(individual,
                                                         integral_experiment=True)
                self.wait_on_jobs('mcnp')

        ### Pulling evaluation data
        for individual in list_of_individuals:

            ### Returning fake fitness value for debug purposes
            if self.options['fake_fitness_debug'] == True:

                for evaluation_type in evaluation_types:
                    individual.debug_fake_fitness(evaluation_type, individual.acceptable_eigenvalue)

                    if evaluation_type == 'integral_keff':
                        if individual.acceptable_eigenvalue:
                            individual.integral_keff_value = individual.integral_keff
                            individual.integral_keff = abs(float(individual.keff) - float(individual.integral_keff_value))
                        else:
                            individual.integral_keff_value = 0.0
                            individual.integral_keff = 0.0
                #print("Faking individual fitness:",individual.ind_count,individual.keff,individual.representativity,individual.total_flux, individual.integral_keff)
            else:
                for evaluation_type in evaluation_types:
                    if evaluation_type == 'representativity':
                        if individual.acceptable_eigenvalue == True:
                            individual.flux_values, individual.flux_uncertainty, individual.total_flux, individual.total_flux_unc = self.mcnp_file_handler.get_f4_flux_from_output(
                                individual.input_file_string + "o")
                            individual.representativity = self.mcnp_file_handler.calculate_representativity_v3(
                                individual.flux_values, self.options['energy_bins'])
                        if individual.acceptable_eigenvalue == False:
                            individual.representativity = 0.0

                    if evaluation_type == 'total_flux':
                        if individual.acceptable_eigenvalue == True:
                            individual.flux_values, individual.flux_uncertainty, individual.total_flux, individual.total_flux_unc = self.mcnp_file_handler.get_f4_flux_from_output(
                                individual.input_file_string + "o")
                            individual.representativity = self.mcnp_file_handler.calculate_representativity_v3(
                                individual.flux_values, self.options['energy_bins'])
                        if individual.acceptable_eigenvalue == False:
                            individual.representativity = 0.0
                            individual.total_flux = 0.0

                    if evaluation_type == 'integral_keff':
                        if individual.acceptable_eigenvalue == True:
                            individual.integral_keff_value = self.mcnp_file_handler.get_keff(individual.integral_keff_input_file_string)
                            ### Todo: Verify that taking the absolute value is correct for integral exp. score. Do we want experiments that add reactivity?
                            individual.integral_keff = abs(float(individual.keff) - float(individual.integral_keff_value))

                        if individual.acceptable_eigenvalue == False:
                            individual.integral_keff = 0.0
                            individual.integral_keff_value = 0.0

            #if self.options['fake_fitness_debug'] == True:
            #    individual.representativity = 0.0
            #    individual.total_flux = 0.0
            #    if individual.keff < individual.options['enforced_maximum_eigenvalue']:
            #        individual.representativity = random.uniform(0, 1.0)
            #        individual.total_flux = random.uniform(0, 1.0)
            #else:
            #    if individual.acceptable_eigenvalue == True:
            #        individual.flux_values, individual.flux_uncertainty, individual.total_flux, individual.total_flux_unc = self.mcnp_file_handler.get_f4_flux_from_output(individual.input_file_string + "o")
            #        individual.representativity = self.mcnp_file_handler.calculate_representativity_v2(individual.flux_values)
             #   if individual.acceptable_eigenvalue == False:
             #       individual.representativity = 0.0
             #       individual.total_flux = 0.0

            #if 'cnn' in self.options['solver']:
            #    print("solving for k with cnn")
            #    self.create_cnn_input()
            #    self.solve_for_keff_with_cnn()

        for constraint in self.options['constraint']:
            if 'postevaluate' in constraint:
                list_of_individuals = self.apply_constraint(list_of_individuals, constraint)

        return list_of_individuals
    #def create_cnn_input(self):
    #    data_array = self.cnn_handler.build_individuals_array(self.individuals, generation=self.generation)

    #    self.cnn_input = self.cnn_handler.build_input_data_reshaped(data_array,
    #                                                                X_val=11,
    #                                                                Y_val=11,
    #                                                                Z_val=1,
    #                                                                channels=2)
        # print("HERE!!! create_cnn_input")

    #def solve_for_keff_with_cnn(self):
    #    self.cnn_predictions = self.cnn_handler.model.predict(self.cnn_input)
    #    # print("PREDICTIONSSSSS", self.cnn_predictions, len(self.cnn_input))
    #    for pred_count, prediction in enumerate(self.cnn_predictions):
    #        # print("PREDICTION", prediction)

    #        self.individuals[pred_count].keff = prediction[0]


    def check_number_of_inds_meeting_constraint(self, checking_value = 'acceptable_eigenvalue'):
        ### Checking to see if any individuals have met keff constraint, if not, sorting by keff
        number_of_acceptable_individuals = 0
        for ind in self.parents_list:
            if getattr(ind, checking_value):
                number_of_acceptable_individuals += 1
        print(number_of_acceptable_individuals, "meet", checking_value)
        return number_of_acceptable_individuals

    ### The crossover function creates total population - number of parents
    def crossover(self):

        if self.options['use_non_dominated_sorting']:
            self.parents_list = self.non_dominated_sorting()
            ### TODO: check on keff constraint more general, right now it is hardcoded
            if self.check_number_of_inds_meeting_constraint(checking_value='acceptable_eigenvalue') == 0:
                self.parents_list.sort(key=lambda x: x.keff, reverse=False)
        else:
            if self.check_number_of_inds_meeting_constraint(checking_value = 'acceptable_eigenvalue') > 0:
                self.individuals.sort(key=lambda x: getattr(x, self.options['fitness_sort_by']), reverse=True)
            else:
                self.individuals.sort(key=lambda x: x.keff, reverse=False)
            ### Pairing down individuals to be specified number
            self.individuals = self.individuals[:self.options['number_of_individuals']]
            self.parents_list = self.individuals[:self.options['number_of_parents']]


        ### Evaluating diversity of population
        if self.options['choose_parent_based_on_bitwise_diversity']:
            #print("Evaluating diversity of parents")
            self.evaluate_bitwise_diversity_of_parents()

        number_of_children = self.options['number_of_individuals'] - \
                             self.options['number_of_parents']
        list_of_children = []
        for new_child_value in range(number_of_children):
            ### Getting parent unique parent number
            parent_1 = random.randint(0, self.options['number_of_parents'] - 1)
            parent_2 = random.randint(0, self.options['number_of_parents'] - 1)
            ### Ensuring unique parents
            while parent_1 == parent_2:
                parent_2 = random.randint(0, self.options['number_of_parents'] - 1)

            ### Using bitwise diversity to choose parent, overwriting previously choosen par 2
            if self.options['choose_parent_based_on_bitwise_diversity']:
                # print("Choosing parent 2 based on diversity score")
                parent_2 = self.choose_parent_based_on_bitwise_diversity(parent_1)

            ### Getting parent individual objects
            parent_1 = self.parents_list[parent_1]
            parent_2 = self.parents_list[parent_2]

            ### Doing bitwise cross over: Child is made by 50/50 chance from each parent
            if self.options['crossover_type'] == 'bitwise':
                ### Creating child individual
                child_ind = individual.individual(self.options, self.generation, self.individual_count)
                self.individual_count += 1

                new_child_ind = self.bitwise_crossover(parent_1, parent_2, child_ind, material_matrix_value = "material_matrix")

                if self.options['adjustable_zone_2A_cassette_bool']:
                    new_child_ind = self.bitwise_crossover(parent_1, parent_2, new_child_ind, material_matrix_value = "material_matrix_cassette_2A")


                ### Checking if new child meets fuel # requirement, remaking if needed
                if self.options['verify_fuel_mass_after_crossover']:
                    fuel_count = new_child_ind.count_material(1)
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        ### Remaking child, but not updating individual count
                        child_ind = individual.individual(self.options, self.generation, self.individual_count)
                        new_child_ind = self.bitwise_crossover(parent_1, parent_2, child_ind,  material_matrix_value="material_matrix_cassette_2A")

                        ### Doing cassetta 2a crossover
                        if self.options['adjustable_zone_2A_cassette_bool']:
                            new_child_ind = self.bitwise_crossover(parent_1, parent_2, new_child_ind,
                                                                   material_matrix_value="material_matrix_cassette_2A")



                        fuel_count = new_child_ind.count_material(1)

            if self.options['crossover_type'] == 'singlepoint':
                new_child_ind = self.singlepoint_crossover(parent_1, parent_2)
                if self.options['verify_fuel_mass_after_crossover']:
                    ### Checking if new child meets fuel # requirement
                    fuel_count = new_child_ind.count_material(1)
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        new_child_ind = self.singlepoint_crossover(parent_1, parent_2)
                        fuel_count = new_child_ind.count_material(1)

            try:
                new_child_ind.parent_string = parent_1.scale_input_filename + "," + parent_2.scale_input_filename
            except:
                new_child_ind.parent_string = str(parent_1.ind_count) + "," + str(parent_2.ind_count)

            new_child_ind.born_from_crossover = True

            list_of_children.append(new_child_ind)

        return list_of_children

    def build_and_run_mcnp_evaluation_input(self, individual,
                                integral_experiment=False):

        ### Setting template file and input file string based on type of evaluation
        if integral_experiment:
            template_file_ = self.options['mcnp_keff_exp_template_file_string']
            input_file_string_ = individual.keff_input_file_string.split('.inp')
            input_file_string = input_file_string_[0] + "_integral_exp.inp"
            individual.integral_keff_input_file_string = input_file_string
        else:
            template_file_ = self.options['mcnp_template_file_string']
            input_file_string = individual.input_file_string

        ### Building dictionary of replacements for the MCNP file
        data_dictionary_ = individual.create_discrete_material_mcnp_dictionary(
            material_matrix_val='material_matrix',
            keywords_list=self.options['keywords_list'])

        ### If doing a variable cassette 2A, adding those values to the data_dictionary_
        if self.options['adjustable_zone_2A_cassette_bool']:
            data_dictionary_ = individual.create_discrete_material_mcnp_dictionary(
                material_matrix_val='material_matrix_cassette_2A',
                keywords_list=self.options['adjustable_zone_2A_keywords_list'],
                data_dict=data_dictionary_)

            data_dictionary_ = individual.build_variable_cassette_2a_dictionary(data_dictionary_)

        ### Building MCNP input file
        self.mcnp_file_handler.write_mcnp_input(template_file=template_file_,
                                                dictionary_of_replacements=data_dictionary_,
                                                input_file_str=input_file_string)
        self.mcnp_file_handler.build_mcnp_running_script(input_file_string)

        self.mcnp_file_handler.run_mcnp_input(input_file_string)
        self.mcnp_inputs.append(input_file_string)
        return

    def bitwise_crossover(self, parent_1, parent_2, child_ind, material_matrix_value = 'material_matrix'):

        print("parent 1 pattern:", material_matrix_value, getattr(parent_1, material_matrix_value), len(getattr(parent_1, material_matrix_value)))
        print("parent 2 pattern:", material_matrix_value, getattr(parent_2, material_matrix_value), len(getattr(parent_2, material_matrix_value)))
        print("Child pattern before:", getattr(child_ind, material_matrix_value), child_ind.ind_count)

        ### Choosing either parent's length
        parent_selection = random.randint(0, 1)

        material_matrix_ = getattr(parent_1, material_matrix_value)
        if parent_selection == 1:
            material_matrix_ = getattr(parent_2, material_matrix_value)

        temp_material_master_list = []
        #for material_list_count, material_list in enumerate(parent_1.material_matrix):
        for material_list_count, material_list in enumerate(material_matrix_):
            temp_material_list = []
            for material_count, material in enumerate(material_list):
                selection = random.randint(0, 1)

                ### Tries to get parent_1 value, if it cant (the list is too short), defaults to parent 2
                try:
                    material = getattr(parent_1, material_matrix_value)[material_list_count][material_count]
                except:
                    material = getattr(parent_2, material_matrix_value)[material_list_count][material_count]

                ### Tries to get parent_2 value, if it cant (the list is too short), defaults to parent 1
                if selection == 1:
                    try:
                        material = getattr(parent_2, material_matrix_value)[material_list_count][material]
                    except:
                        material = getattr(parent_1, material_matrix_value)[material_list_count][material_count]

                temp_material_list.append(material)
            temp_material_master_list.append(temp_material_list)
        setattr(child_ind, material_matrix_value, temp_material_master_list)

        ### Else, the material matrices are the same length, so they can be crossed over no problem.
        ### TODO: This loop may be unneeded, the above loop likely would cover this case as well, right?
        '''else:
            temp_material_master_list = []
            #for material_list_count, material_list in enumerate(parent_1.material_matrix):
            for material_list_count, material_list in enumerate(getattr(parent_1, material_matrix_value)):
                temp_material_list = []
                for material_count, material in enumerate(material_list):
                    selection = random.randint(0, 1)

                    material = getattr(parent_1, material_matrix_value)[material_list_count][material_count]
                    if selection == 1:
                        material = getattr(parent_2, material_matrix_value)[material_list_count][material]

                    temp_material_list.append(material)
                temp_material_master_list.append(temp_material_list)
            setattr(child_ind, material_matrix_value, temp_material_master_list)'''

        print("Child pattern after:", material_matrix_value, getattr(child_ind, material_matrix_value), len(getattr(child_ind, material_matrix_value)), child_ind.ind_count)
        return child_ind

    def singlepoint_crossover(self, parent_1, parent_2):
        child_ind = individual.individual(self.options, self.generation)

        temp_material_master_list = []
        for material_list_count, material_list in enumerate(child_ind.material_matrix):
            temp_material_list = []

            ml_length = len(material_list) - 1

            singlepoint_value = random.randint(1, ml_length)

            temp_material_list = parent_1.material_matrix[material_list_count][0:singlepoint_value] + \
                                 parent_2.material_matrix[material_list_count][singlepoint_value - 1:ml_length]

            temp_material_master_list.append(temp_material_list)
        child_ind.material_matrix = temp_material_master_list

        return child_ind

    def variable_cassette_length_mutation(self, original_material_matrix, mutating_debug = False):
        new_material_matrix = []
        original_mm_length_int = len(original_material_matrix)
        valid_mutation = False

        if mutating_debug:
            print("Doing variable cassette length mutation. Original length: {}".format(original_mm_length_int))

        for count, value in enumerate(original_material_matrix):
            add_plate = True
            ### Catch if trying something more than 1D
            assert len(value) == 1, "Unable to do length mutation on geometries more complex than 1D"
            if mutating_debug:
                print("Attempting mutation on plate #{}".format(count))
            ### If there hasn't yet been a valid mutation, doing mutation loop.
            if valid_mutation == False:
                ### Throwing a random number to see if variable will change
                if random.uniform(0, 1.0) <= self.options['chance_for_cassette_length_mutation']:
                    if mutating_debug:
                        print("There's a length mutation!")
                    ### A mutation happens. Throwing another random value to see if a plate is added or removed
                    if random.uniform(0, 1.0) <= self.options['chance_for_smaller_cassette_length_mutation']:
                        if mutating_debug:
                            print("The cassette will be made smaller! Skipping plate #{}".format(count))
                        ### The material matrix is going to be 1 plate smaller, not adding the current plate to
                        ### the new list
                        add_plate = False
                        valid_mutation = True
                    else:
                        if mutating_debug:
                            print("The cassette will be made larger!")
                        ### If there are already the maximum number of plates, continuing
                        if original_mm_length_int == self.options['adjustable_zone_2A_fixed_values']['maximum_plates']:
                            if mutating_debug:
                                print("There are already a maximum number of plates, cannot add one.")
                            continue
                        else:
                            random_new_material = random.randint(0, len(self.options['adjustable_zone_2A_material_types']))
                            new_material_matrix.append([self.options['adjustable_zone_2A_material_types'][random_new_material]])
                            if mutating_debug:
                                print("Added plate type {} to location {}".format(self.options['adjustable_zone_2A_material_types'][random_new_material], count))
                            valid_mutation = True
            else:
                if mutating_debug:
                    print("A valid mutation has already been done, adding original material to matrix")
            if add_plate:
                new_material_matrix.append(value)

        if mutating_debug:
            print("original matrix:", original_material_matrix)
            print("new matrix:", new_material_matrix)
            print("Finished mutation function. New matrix length: {}".format(len(new_material_matrix)))

        return new_material_matrix



    def mutate(self, list_of_individuals, material_matrix_value, do_variable_size_mutation = False, mutating_debug = False):
        ### Currently only works on a material-basis
        if mutating_debug:
            print("MUTATING!!!")
        if self.options['mutation_type'] == 'bitwise':
            if mutating_debug:
                print("BITWISE!", len(list_of_individuals))
            for ind_count, individual in enumerate(list_of_individuals):
                if mutating_debug:
                    print("MUTATING:", ind_count)

                ### Preserving an original copy of the material matrix
                original_material_matrix = copy.deepcopy(getattr(individual, material_matrix_value))

                ### Doing single bit mutation
                setattr(individual, material_matrix_value, self.single_bit_mutation(original_material_matrix))

                ### If the cassette is a variable length, doing that mutation
                if do_variable_size_mutation:
                    setattr(individual, material_matrix_value, self.variable_cassette_length_mutation(original_material_matrix, mutating_debug))

                    ### Resetting the number of plates in this individual to the correct value
                    setattr(individual, "number_of_plates_in_cassette_2A", len(getattr(individual, material_matrix_value)))



                if self.options['verify_fuel_mass_after_mutation']:
                    ### Checking if new child meets fuel # requirement
                    fuel_count = individual.count_material(self.options['fuel_material_number'])
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        setattr(individual, material_matrix_value, self.single_bit_mutation(original_material_matrix))
                        fuel_count = individual.count_material(self.options['fuel_material_number'])

        return list_of_individuals

    def single_bit_mutation(self, material_matrix, force_mutation = False, force_mutation_per_material_sublist = 1):
        new_material_matrix = []
        #print("old material matrix:", material_matrix)
        ### If forcing a certain number of mutations per material sublist:
        force_mutation_index = []
        if force_mutation:
            for _ in range(force_mutation_per_material_sublist):
                rand_val = random.randint(0, len(material_matrix) - 1)
                while rand_val in force_mutation_index:
                    rand_val = random.randint(0, len(material_matrix) - 1)
                    #print(rand_val, force_mutation_index)
                force_mutation_index.append(rand_val)
            #print("FORCING A MUTATION! at index(es):", force_mutation_index)

        for material_list in material_matrix:
            material_matrix_sublist = []



            for material_count, material in enumerate(material_list):
                ### Attempting mutation
                random_val = random.uniform(0, 1.0)

                ### If the current material count is in the force mutation list, forcing the mutation
                if force_mutation:
                    if material_count in force_mutation_index:
                        random_val = self.options['mutation_rate']
                        #print("Forcing a mutation in material:", material_count)


                if random_val <= self.options['mutation_rate']:

                    new_index = random.randint(0, len(self.options['material_types']) - 1)
                    while material == self.options['material_types'][new_index]:
                        new_index = random.randint(0, len(self.options['material_types']) - 1)
                    # print("NEW_INDEX: ", new_index, len(self.options['material_types']) - 1)
                    # print("new material: ", self.options['material_types'][new_index], "old", material)
                    material = self.options['material_types'][new_index]
                material_matrix_sublist.append(material)
            new_material_matrix.append(material_matrix_sublist)
        #print("new_material_matrix:", new_material_matrix)

        return new_material_matrix

    def submit_jobs(self, job_list):
        for job in job_list:
            print("Submitting job:", job)
            os.system('qsub ' + job)

    def wait_on_jobs(self, run_type, unique_flag = ""):
        waiting_on_jobs = True
        jobs_completed = 0
        total_time = 0
        jobs_to_be_waited_on = getattr(self, run_type + "_inputs")
        temp_file_list = copy.deepcopy(jobs_to_be_waited_on)
        print("Jobs waiting on: ", jobs_to_be_waited_on)
        while waiting_on_jobs:
            for file in os.listdir():
                if "gen_" + str(self.generation) in file:
                    if "_done.dat" in file:
                        #if unique_flag in file:
                        file_temp = file.split("_done.dat")
                        script_str = file_temp[0]
                        if script_str in temp_file_list:
                            temp_file_list.remove(file_temp[0])
                            jobs_completed += 1
            if jobs_completed == len(jobs_to_be_waited_on):
                print("All jobs are complete, continuing")
                return

            print("Jobs Complete: ", jobs_completed, "Jobs pending:", len(jobs_to_be_waited_on) - jobs_completed)
            for file in temp_file_list:
                print(file)
            if self.options['skip_waiting_on_jobs_debug']:
                print("""options['skip_waiting_on_jobs_debug'] is True, continuing""")
                return
            print("Waiting 15 seconds. Total time (mins):", str(total_time/60))
            time.sleep(15)
            total_time += 15

    def enforce_fuel_count(self):
        for individual in self.individuals:
            fuel_count = individual.count_material(1)
            if fuel_count != self.options['enforced_fuel_count_value']:
                individual.fix_material_count(1, self.options['enforced_fuel_count_value'])

    def write_output(self):
        output_file = open(self.options['output_filename'] + '.csv', 'a')

        ###Building string to write
        number_of_children_needed = self.options['number_of_individuals'] - self.options['number_of_parents']
        number_of_inds_from_current_generation = 0
        for ind_count, individual in enumerate(self.individuals):
            if ind_count <= self.options['number_of_parents'] - 1:
                write_string = self.write_options_funct(output_file, individual)
                output_file.write(write_string + "\n")
                if individual.generation == self.generation:
                    number_of_inds_from_current_generation += 1

                continue
            if individual.generation == self.generation:
                write_string = self.write_options_funct(output_file, individual)
                output_file.write(write_string + "\n")
                if individual.generation == self.generation:
                    number_of_inds_from_current_generation += 1
                continue

            if ind_count < self.options['number_of_individuals'] - 1:
                if number_of_children_needed > number_of_inds_from_current_generation:
                    continue
                write_string = self.write_options_funct(output_file, individual)
                output_file.write(write_string + "\n")
                continue
        output_file.close()

    def write_output_v2(self, list_of_individuals, output_file_name = ""):

        ### Setting the default value, for the by-generation output
        if output_file_name == "":
            output_file_name = self.options['output_filename'] + '.csv'

        if output_file_name.endswith('.csv') != True:
            output_file_name += ".csv"


        output_file = open(output_file_name, 'a')

        ###Building string to write
        for ind_count, individual in enumerate(list_of_individuals):
            #print(ind_count, individual.input_file_string)
            write_string = self.write_options_funct(output_file, individual)
            # print("writing out child:", self.generation, ind_count, write_string)
            output_file.write(write_string + "\n")

        output_file.close()

    def write_options_funct(self, output_file, individual):
        write_string = ""
        for write_option in self.options['output_writeout_values']:
            if "#" in write_option:
                write_option_split = write_option.split("#")
                write_option = write_option_split[0]
            if write_option == 'generation':
                write_string += str(self.generation) + ","
            if write_option == 'individual_count':
                write_string += str(individual.ind_count) + ","
            if write_option == 'keff':
                write_string += str(individual.keff) + ","
            if write_option == 'front_rank':
                try:
                    write_string += str(individual.front_rank) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'crowding_distance':
                try:
                    write_string += str(individual.crowding_distance) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'integral_keff':
                try:
                    write_string += str(individual.integral_keff) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'integral_keff_value':
                try:
                    write_string += str(individual.integral_keff_value) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'total_flux':
                try:
                    write_string += str(individual.total_flux) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'representativity':
                try:
                    write_string += str(individual.representativity) + ","
                except:
                    write_string += "N/A,"

            if write_option == 'number_of_plates_in_cassette_2A':
                try:
                    write_string += str(individual.number_of_plates_in_cassette_2A) + ","
                except:
                    write_string += "N/A,"
            if write_option == 'zone_2A_materials':
                write_string += \
                    str(individual.make_material_string_csv('material_matrix_cassette_2A',
                                                            'material_matrix_cassette_2A_str'))
            if write_option == 'materials':
                write_string += str(individual.make_material_string_csv('material_matrix', 'material_string_csv'))
            if write_option == 'input_name':
                try:
                    write_string += str(individual.input_file_string) + ','
                except:
                    write_string += "N/A,"
            if write_option == 'number_of_fuel':
                write_string += str(individual.count_material(1)) + ','
            if write_option == 'write_out_parents':
                write_string += str(individual.parent_string) + ','
            if write_option == 'write_out_average_diversity_score':
                if self.options['choose_parent_based_on_bitwise_diversity'] == True:
                    try:
                        average_tds = individual.total_diversity_score / (self.options['number_of_parents'] - 1)
                        write_string += str(average_tds) + ','
                    except:
                        write_string += "N/A,"
                else:
                    write_string += "N/A,"
        return write_string

    def create_header(self):
        header_string = ""
        for val in self.options['output_writeout_values']:
            if "#" in val:
                val_split = val.split("#")
                val = val_split[0]
                val_range = int(val_split[1])
                for _ in range(val_range):
                    header_string += val + str(_)+ ","
            else:
                header_string += val + ","

        return header_string
