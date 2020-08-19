import os
import collections
import math
import random
import copy
import time
random.seed(865)

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



        self.generation = 0
        self.individual_count = 0
        ### Creating initial population
        for ind in range(self.options['number_of_individuals']):
            self.individuals.append(individual.individual(options_dict, self.generation, self.individual_count))
            self.individual_count += 1

        if self.options['remake_duplicate_children'] == True:
            self.all_individuals = copy.deepcopy(self.individuals)
            print("All individuals:", self.all_individuals)
        ### Loading CNN if needed
        #if 'cnn' in self.options['solver']:
        #    model_string = "CNN_3d_11x11_fm_cad_4x4_kern_v2.hdf5"
        #    self.cnn_handler = CNN_Handler.CNN_handler(model_string)
        #    self.cnn_input = []

        if self.options['include_pattern']:
            print("Including a pattern in initial population!")
            for ind_count, pattern_to_include in enumerate(self.options['pattern_to_include']):
                for _ in self.individuals[ind_count].material_matrix:
                    print(_)
                self.individuals[ind_count].material_matrix = pattern_to_include
                self.individuals[ind_count].make_material_string_scale('%array%1')
                for _ in self.individuals[ind_count].material_matrix:
                    print(_)

        if self.options['enforce_fuel_count']:
            print("enforcing material",self.options['enforce_material_number'],' count:', self.options['enforced_fuel_count_value'])
            for ind_count, ind in enumerate(self.individuals):
                ind.enforce_material_count(1, self.options['enforced_fuel_count_value'])

        ### Creating output csv if needed
        if self.options['write_output_csv']:
            output_csv = open(self.options['output_filename'] + '.csv', 'w')
            for flag in self.options:
                output_csv.write("{},{}\n".format(flag, self.options[flag]))
            output_csv.close()


        ### Running eigenvalue calcs if needed
        #if self.options['enforced_maximum_eigenvalue'] == True:
        #    getattr(self, self.options['check_eigenvalue_function'])()

        ### Evaluating initial population, gen 0
        print("Evaluating initial population")
        self.evaluate(self.options['fitness'])

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

        self.write_output_v2(self.parents_list)
        self.generation += 1

        ### Running GA algo
        for generation in range(self.options['number_of_generations']):
            print("Generation: ", self.generation)
            print("crossover")
            list_of_children = self.crossover()
            print("mutating")
            list_of_mutated_children = self.mutate(list_of_children)

            if self.options['remake_duplicate_children'] == True:
                list_of_mutated_children = self.remake_duplicate_children(list_of_mutated_children, self.all_individuals)
                self.all_individuals += list_of_mutated_children

            if self.options['enforce_fuel_count']:
                print("enforcing fuel count:", self.options['enforced_fuel_count_value'])
                for ind_count, ind in enumerate(list_of_mutated_children):
                    ind.enforce_material_count(1, self.options['enforced_fuel_count_value'])

            print("evaluating children")
            self.evaluate(self.options['fitness'], list_of_mutated_children)
            #for ind_count, ind_ in enumerate(list_of_mutated_children):
            #    print(ind_count, ind_.ind_count, ind_.generation, ind_.representativity)
            #print("CHILDREN:::")
            #for ind_count, ind_ in enumerate(list_of_mutated_children):
                #print(ind_count, ind_.ind_count, ind_.generation, ind_.representativity)

            ### Checking if any of the children have already been created/evaluated

            ### combining now evaluated children with previous list of individuals
            self.individuals = self.parents_list + list_of_mutated_children




            print("All individuals in this generation!")
            for ind_count, ind_ in enumerate(self.individuals):
                print(ind_count, ind_.ind_count, ind_.generation, ind_.keff, ind_.representativity)

            print("write output")
            if self.options['write_output_csv']:
                self.write_output_v2(self.parents_list)

            self.generation += 1

            if self.options['remake_duplicate_children'] == True:
                self.all_individuals += list_of_mutated_children

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
                    data_dictionary_ = individual.create_discrete_material_mcnp_dictionary(
                        self.options['keywords_list'])
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
                        ind.keff = random.uniform(0.5, 1.5)
                    else:
                        ind.keff = self.mcnp_file_handler.get_keff(ind.keff_input_file_string)

                    ### If that keff is above a set threshold, sets acceptable_eigenvalue to false. Else, sets it True.
                    if float(ind.keff) >= self.options['enforced_maximum_eigenvalue']:
                        print("keff, ", ind.keff, "too high. Skipping source calculation")
                        ind.acceptable_eigenvalue = False
                        ind.keff = 0.0
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
        return list_of_individuals

    def non_dominated_sorting(self):
        ###
        front = [[]]
        for individual in self.individuals:
            individual.front_rank = 'none'
            print("Nondominated sorting", individual.input_file_string, individual.keff, individual.representativity)
            ### Initializing lists
            dominated_list = []
            number_of_inds_that_dominate_individual = 0
            #for fitness_ in self.options['fitness']:
            #dominated_list.append([])
            #number_of_inds_that_dominate_individual.append(0)

            ### Iterating over all individuals, comparing fitnesses
            for individual_ in self.individuals:
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
            #print(individual.input_file_string, individual.keff, individual.representativity, len(individual.dominated_list), individual.number_of_inds_that_dominate_individual)

            #for fitness_index, fitness_ in enumerate(self.options['fitness']):
            if individual.number_of_inds_that_dominate_individual == 0:
                individual.front_rank = 0
                print("Ind is rank one:", individual.input_file_string, individual.representativity, individual.keff)
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
                    #print(individual_.input_file_string, individual_.front_rank, current_front)

            front.append(current_front)
            pareto_front += 1
        #print("Pareto fronts:")
        #for front_count, front_list in enumerate(front):
            #print(front_list)
            #for ind in front_list:
                #print(front_count, ind.representativity, ind.keff)
        self.pareto_front = front

        #for individual_ in self.individuals:
            #print(individual_.input_file_string,  individual_.representativity, individual_.keff)


        ### Building list of parents
        parents_list = []
        for front in self.pareto_front:
            if front == []:
                continue
            print("Front:", len(front))
            if len(front) < (self.options['number_of_parents'] - len(parents_list)):
                print("Length of parent list!!!", len(parents_list))
                parents_list = parents_list + front
                print("Length of parent list!!!", len(parents_list))
            else:

                front = self.crowding_distance(front)
                print(len(front), self.options['number_of_parents'], len(parents_list))
                front.sort(key=lambda x: x.crowding_distance, reverse=True)

                ind_count = 0
                print("Adding parents to parents list")
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
                except:
                    continue
                #print("CROWDING DISTANCE!!!", ind.crowding_distance, count, len(front))

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



    def remake_duplicate_children(self, list_of_children, comparison_list):

        for child in list_of_children:
            print("Checking child:", child.material_matrix, comparison_list)
            for comparison_ind in comparison_list:
                #print("comparison", child.material_matrix, comparison_ind.material_matrix)
                comparison_score = 0
                for child_mat, comp_mat in zip(child.material_matrix, comparison_ind.material_matrix):
                    if child_mat == comp_mat:
                        comparison_score += 1
                    if comparison_score == self.options['total_materials']:
                        print("Duplicate child found! Forcing mutation")
                        print("Before", child.material_matrix)
                        child.material_matrix = self.single_bit_mutation(child.material_matrix, force_mutation=True, force_mutation_per_material_sublist=2)
                        print("After", child.material_matrix)
                #print(child.material_matrix, comparison_ind.material_matrix)
                #if child.material_matrix == comparison_ind.material_matrix:
                #    child.create_random_pattern()
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
            if 'evaluate' in constraint:
                list_of_individuals = self.apply_constraint(list_of_individuals, constraint)

        for evaluation_type in evaluation_types:
            if "#" in evaluation_type:
                evaluation_type, evaluation_options = evaluation_type.split("#")
            scale_inputs = []
            if evaluation_type == 'representativity':
                print("Evaluating Representativity")
                if 'mcnp' in self.options['solver']:
                    self.mcnp_inputs = []
                    for individual in list_of_individuals:
                        if individual.ran_source_calculation == False:
                            if individual.acceptable_eigenvalue == True:
                                ### Building MCNP input file
                                self.build_mcnp_source_input(individual)
                                individual.ran_source_calculations = True
                    self.wait_on_jobs('mcnp')

                    for individual in list_of_individuals:
                        if self.options['fake_fitness_debug'] == True:
                            individual.representativity = random.uniform(0, 1.0)
                        else:
                            if individual.acceptable_eigenvalue == True:
                                current_vals, current_unc = self.mcnp_file_handler.get_flux(individual.input_file_string + "o")
                                individual.representativity = self.mcnp_file_handler.calculate_representativity(current_vals, current_unc)
                            if individual.acceptable_eigenvalue == False:
                                individual.representativity = 0.0

                        print("individual.representativity", individual.representativity)

            if evaluation_type == 'total_flux':
                print("Evaluating Total Flux")

                for individual in list_of_individuals:
                    if self.options['fake_fitness_debug'] == True:
                        individual.total_flux = random.uniform(0, 1.0)
                    else:
                        if individual.acceptable_eigenvalue == True:
                            individual.flux_values, individual.flux_uncertainty = self.mcnp_file_handler.get_flux(individual.input_file_string + "o")
                            individual.total_flux = individual.flux_values[-1]
                        if individual.acceptable_eigenvalue == False:
                            individual.total_flux = 0.0

                    print("individual.tota_flux", individual.total_flux)



            #if 'cnn' in self.options['solver']:
            #    print("solving for k with cnn")
            #    self.create_cnn_input()
            #    self.solve_for_keff_with_cnn()
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

    ### The crossover function creates total population - number of parents
    def crossover(self):

        if self.options['use_non_dominated_sorting'] == True:
            self.parents_list = self.non_dominated_sorting()
        else:
            self.individuals.sort(key=lambda x: getattr(x, self.options['fitness_sort_by']), reverse=True)
            ### Pairing down individuals to be specified number
            self.individuals = self.individuals[:self.options['number_of_individuals']]
            self.parents_list = self.individuals[:10]

        ### Evaluating diversity of population
        if self.options['choose_parent_based_on_bitwise_diversity']:
            print("Evaluating diversity of parents")
            self.evaluate_bitwise_diversity_of_parents()

        number_of_children = self.options['number_of_individuals'] - \
                             self.options['number_of_parents']
        list_of_children = []
        for new_child_value in range(number_of_children):
            ### Getting parent values
            parent_1 = random.randint(0, self.options['number_of_parents'] - 1)
            parent_2 = random.randint(0, self.options['number_of_parents'] - 1)
            while parent_1 == parent_2:
                parent_2 = random.randint(0, self.options['number_of_parents'] - 1)

            if self.options['choose_parent_based_on_bitwise_diversity']:
                # print("Choosing parent 2 based on diversity score")
                parent_2 = self.choose_parent_based_on_bitwise_diversity(parent_1)

            parent_1 = self.parents_list[parent_1]
            parent_2 = self.parents_list[parent_2]
            if self.options['crossover_type'] == 'bitwise':
                new_child_ind = self.bitwise_crossover(parent_1, parent_2)

                ### Checking if new child meets fuel # requirement
                if self.options['verify_fuel_mass_after_crossover']:
                    fuel_count = new_child_ind.count_material(1)
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        new_child_ind = self.bitwise_crossover(parent_1, parent_2)
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

    def build_mcnp_source_input(self, individual_):
        ### Building MCNP input file
        self.mcnp_file_handler.write_mcnp_input(template_file=self.options['mcnp_template_file_string'],
                                                dictionary_of_replacements=individual_.create_discrete_material_mcnp_dictionary(
                                                    self.options['keywords_list']),
                                                input_file_str=individual_.input_file_string)
        self.mcnp_file_handler.build_mcnp_running_script(individual_.input_file_string)

        self.mcnp_file_handler.run_mcnp_input(individual_.input_file_string)
        self.mcnp_inputs.append(individual_.input_file_string)
        return

    def bitwise_crossover(self, parent_1, parent_2):
        child_ind = individual.individual(self.options, self.generation, self.individual_count)
        self.individual_count += 1
        # print("parent 1 pattern:", parent_1.material_matrix)
        # print("parent 2 pattern:", parent_2.material_matrix)
        # print("Child pattern before:", child_ind.material_matrix, child_ind.ind_count)
        temp_material_master_list = []
        for material_list_count, material_list in enumerate(parent_1.material_matrix):
            temp_material_list = []
            for material_count, material in enumerate(material_list):
                selection = random.randint(0, 1)

                material = parent_1.material_matrix[material_list_count][material_count]

                if selection == 1:
                    material = parent_2.material_matrix[material_list_count][material_count]

                temp_material_list.append(material)
            temp_material_master_list.append(temp_material_list)
        child_ind.material_matrix = temp_material_master_list
        # print("Child pattern after:", child_ind.material_matrix)
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

    def mutate(self, list_of_individuals):
        ### Currently only works on a material-basis
        print("MUTATING!!!")
        if self.options['mutation_type'] == 'bitwise':
            print("BITWISE!", len(list_of_individuals))
            for ind_count, individual in enumerate(list_of_individuals):
                print("MUTATING:", ind_count)
                original_material_matrix = copy.deepcopy(individual.material_matrix)
                individual.material_matrix = self.single_bit_mutation(original_material_matrix)

                if self.options['verify_fuel_mass_after_mutation']:
                    ### Checking if new child meets fuel # requirement
                    fuel_count = individual.count_material(1)
                    # try_count = 0
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        individual.material_matrix = self.single_bit_mutation(original_material_matrix)
                        fuel_count = individual.count_material(1)
                        # print("mutation fuel count:", fuel_count)
                        # try_count  += 1
                    # print("fixed mutation in:", try_count, "tries")

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
                    print(rand_val, force_mutation_index)
                force_mutation_index.append(rand_val)
            print("FORCING A MUTATION! at index(es):", force_mutation_index)

        for material_list in material_matrix:
            material_matrix_sublist = []



            for material_count, material in enumerate(material_list):
                ### Attempting mutation
                random_val = random.uniform(0, 1.0)

                ### If the current material count is in the force mutation list, forcing the mutation
                if force_mutation:
                    if material_count in force_mutation_index:
                        random_val = self.options['mutation_rate']
                        print("Forcing a mutation in material:", material_count)


                if random_val <= self.options['mutation_rate']:

                    new_index = random.randint(0, len(self.options['material_types']) - 1)
                    while material == self.options['material_types'][new_index]:
                        new_index = random.randint(0, len(self.options['material_types']) - 1)
                    # print("NEW_INDEX: ", new_index, len(self.options['material_types']) - 1)
                    # print("new material: ", self.options['material_types'][new_index], "old", material)
                    material = self.options['material_types'][new_index]
                material_matrix_sublist.append(material)
            new_material_matrix.append(material_matrix_sublist)
        print("new_material_matrix:", new_material_matrix)

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
                        if unique_flag in file:
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

            # print("Writing ind", ind_count, self.options['number_of_parents']  - 1)

            if ind_count <= self.options['number_of_parents'] - 1:
                write_string = self.write_options_funct(output_file, individual)
                # print("WRITING PARENTS!", self.generation, ind_count, write_string)
                output_file.write(write_string + "\n")

                if individual.generation == self.generation:
                    number_of_inds_from_current_generation += 1

                continue
            if individual.generation == self.generation:

                write_string = self.write_options_funct(output_file, individual)
                # print("writing out child:", self.generation, ind_count, write_string)
                output_file.write(write_string + "\n")
                if individual.generation == self.generation:
                    number_of_inds_from_current_generation += 1
                continue

            if ind_count < self.options['number_of_individuals'] - 1:
                if number_of_children_needed > number_of_inds_from_current_generation:
                    continue
                write_string = self.write_options_funct(output_file, individual)
                # print("writing out filler:", self.generation, ind_count, write_string)
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
            if write_option == 'materials':
                write_string += str(individual.make_material_string_csv())
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

        return write_string