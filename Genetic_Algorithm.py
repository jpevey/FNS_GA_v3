import os
import collections
import math
import random
import copy
import time
import numpy as np

import Individual_v1 as individual
import MCNP_File_Handler
import CNN_Handler

class genetic_algorithm:
    def __init__(self, options_dict):
        self.mcnp_file_handler = MCNP_File_Handler.mcnp_file_handler()



        print("Initializing GA with:", options_dict)
        self.options = options_dict

        self.cnn_handler = CNN_Handler.CNN_handler()
        #self.cnn_handler.load_cnn_model(self.options['CNN_check_keff_model_string'])

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
            print("Including a set of patterns in initial population!")
            for pattern_per_ind in self.options['pattern_to_include']:
                for ind_count, pattern_to_include in enumerate(self.options[pattern_per_ind]):
                    #print(ind_count, pattern_to_include)
                    #print(getattr(self.individuals[ind_count], pattern_per_ind))
                    setattr(self.individuals[ind_count], pattern_per_ind, pattern_to_include)
                    #print(getattr(self.individuals[ind_count], pattern_per_ind))


                    setattr(self.individuals[ind_count], "number_of_plates_in_cassette_2A",
                            len(getattr(self.individuals[ind_count], "material_matrix_cassette_2A")))





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

        self.write_output_v2(self.individuals)
        self.generation += 1

        ### Running GA algo
        for generation in range(self.options['number_of_generations']):
            print("Generation: ", self.generation)
            for constraint in self.options['constraint']:
                if "linearSimAnneal" in constraint:
                    print("Applying linearSimAnneal to all previous individuals")
                    self.all_individuals = self.apply_constraint(self.all_individuals, constraint)

            print("crossover")
            list_of_children = self.crossover(keff_constraint_string = 'keff')

            print("mutating")
            ### Runs CNN mutation heuristic
            if self.options['CNN_check_mutation_bool']:
                print("Applying CNN Mutation heuristic!")
                list_of_mutated_children = self.apply_cnn_mutation_heuristic(list_of_children)
                number_of_mutated_children_created = len(list_of_mutated_children)
                ### If the list of mutated children is less than the number we need, filling it with original individuals
                if number_of_mutated_children_created < self.options['number_of_children']:
                    number_of_original_children_to_keep = self.options['number_of_children'] - number_of_mutated_children_created
                    #print("adding {} individuals to child list".format(number_of_original_children_to_keep))
                    original_list_of_mutated_children = self.mutate(list_of_children[:number_of_original_children_to_keep], 'material_matrix')
                    #print("length of mutate {} ".format(len(original_list_of_mutated_children)))
                    if self.options['adjustable_zone_2A_cassette_bool']:
                        original_list_of_mutated_children = self.mutate(original_list_of_mutated_children,
                                                               'material_matrix_cassette_2A',
                                                               do_variable_size_mutation=True)

                    #print("length of mutate {} ".format(len(original_list_of_mutated_children)))

                    list_of_mutated_children = list_of_mutated_children + original_list_of_mutated_children

                ### Setting the newly made individuals to input/output/ind count of inds made by crossover
                for ind_og, ind_new in zip(list_of_children, list_of_mutated_children):
                    ind_new.ind_count = ind_og.ind_count
                    ind_new.input_file_string = ind_og.input_file_string
                    ind_new.keff_input_file_string = ind_og.keff_input_file_string
                    ### Original mutation functions


            else:
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

            ### If using keff cnn, adjusting threshold/bias of model with how it did predicting the last generation
            if self.options['CNN_check_keff_adjustable_bias']:
                if self.options['CNN_check_keff_crossover'] or self.options['CNN_check_keff_mutation']:
                    sum_pred_keff = 0.0
                    sum_true_keff = 0.0

                    ### Doing CNN prediction and summing pred and tru keff values
                    for ind_ in self.individuals:
                        ind_.keff_cnn = self.predict_keff_with_cnn(ind_)
                        sum_pred_keff += ind_.keff_cnn
                        sum_true_keff += float(ind_.keff)

                    ### Calculating average
                    average_pred_keff = sum_pred_keff / len(self.individuals)
                    average_true_keff = sum_true_keff / len(self.individuals)


                    print("Updating CNN KEFF bias: Old Value {} New Value {} Avg_pred/Avg_keff {}".format(self.options['CNN_check_keff_threshold'],
                                                                                                          average_pred_keff/average_true_keff * self.options['CNN_check_keff_threshold'],
                                                                                                          average_pred_keff/average_true_keff))
                    self.options['CNN_check_keff_threshold'] += (0.95-average_true_keff)

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

    def build_list_of_fns_patterns_for_cnn_3x3(self, total_list_of_mutated_children):
        list_of_all_children_cnn_inputs = []
        for child_object in total_list_of_mutated_children:
            material_cnn_form_preproccessed = self.cnn_handler.preprocess_variable_cassette_child_object(child_object)
            material_cnn_form, _ = self.cnn_handler.build_3x3_fns_data_variable_cassette_A(single_case=True,
                                                                               single_case_material_matrix=material_cnn_form_preproccessed)
            list_of_all_children_cnn_inputs.append(np.reshape(material_cnn_form, [60, 2, 2, 4]))
        return np.array(list_of_all_children_cnn_inputs)

    def evaluate_cnn_objective_function(self,
                                        model_option_flag,
                                        list_of_objectives,
                                        np_array_of_all_children_inputs,
                                        total_list_of_mutated_children,
                                        output_append_flag = "_cnn"):

        for objective_function in list_of_objectives:
            self.cnn_handler.load_cnn_model(self.options[model_option_flag][objective_function])
            predictions = self.cnn_handler.model.predict(np_array_of_all_children_inputs)
            assert len(total_list_of_mutated_children) == len(predictions), "Some how your list of individuals and predictions is not equal."

            for individual, prediction in zip(total_list_of_mutated_children, predictions):
                setattr(individual,  objective_function + output_append_flag, prediction[0])

        return total_list_of_mutated_children

    def apply_cnn_mutation_heuristic(self, list_of_children):

        ### Creating number of children specified
        total_list_of_mutated_children = []
        for val in range(self.options['CNN_check_mutation_number_of_sets_of_mutants']):
            copy_of_list_of_children = copy.deepcopy(list_of_children)
            original_list_of_mutated_children = self.mutate(copy_of_list_of_children,
                                                            'material_matrix', mutating_debug=False)
            if self.options['adjustable_zone_2A_cassette_bool']:
                original_list_of_mutated_children = self.mutate(original_list_of_mutated_children,
                                                                'material_matrix_cassette_2A',
                                                                do_variable_size_mutation=True, mutating_debug=False)


            total_list_of_mutated_children = total_list_of_mutated_children + original_list_of_mutated_children

        ### Building list of input variables
        np_array_of_all_children_inputs = self.build_list_of_fns_patterns_for_cnn_3x3(total_list_of_mutated_children)

        total_list_of_mutated_children = self.evaluate_cnn_objective_function('CNN_check_mutatant_objective_functins',
                                                                              self.options['CNN_check_mutatant_objective_functins'],
                                                                              np_array_of_all_children_inputs,
                                                                              total_list_of_mutated_children,
                                                                              output_append_flag="_cnn")

        ### For each child, evaluating it for the objective functions
#       for objective_function in :
 #           self.cnn_handler.load_cnn_model(self.options['CNN_check_mutatant_objective_functins'][objective_function])
 #           predictions = self.cnn_handler.model.predict(np_array_of_all_children_inputs)
#
 #           assert len(total_list_of_mutated_children) == len(predictions), "Some how your list of individuals and predictions is not equal."
 #           for individual, prediction in zip(total_list_of_mutated_children, predictions):
  #              setattr(individual,  objective_function + "_cnn", prediction[0])"""

        ### Applies the cnn heuristic constraint, lowering the number of individuals to sort
        ### load keff prediction model and make prediction
        if self.options['CNN_check_mutatant_keff_model_string']  == 'default':
            keff_model_str = self.options['CNN_check_keff_model_string']
        else:
            keff_model_str = self.options['CNN_check_mutatant_keff_model_string']
        self.cnn_handler.load_cnn_model(keff_model_str)
        predictions = self.cnn_handler.model.predict(np_array_of_all_children_inputs)
        ### setting prediction to the individuals
        for individual, prediction in zip(total_list_of_mutated_children, predictions):
            setattr(individual, "keff_cnn", prediction[0])
        print("Length of list of children before keff constraint:", len(total_list_of_mutated_children))
        total_list_of_mutated_children = self.apply_cnn_heuristic_constraint(total_list_of_mutated_children, self.options['CNN_check_mutatant_remove_mutants_by_constraint'])
        print("Length of list of children AFTER keff constraint:", len(total_list_of_mutated_children))

        for child in total_list_of_mutated_children:
            print("child ind predictions", child.keff_input_file_string, child.keff_cnn, child.representativity_cnn, child.total_flux_cnn)

        ### Sorting these individuals
        sorted_list_of_children = self.non_dominated_sorting(individuals_to_sort =total_list_of_mutated_children,
                                                      fitness_to_nds = self.options['CNN_check_mutatant_objective_functin_attrs'],
                                                      number_of_individuals_to_keep = self.options['CNN_check_mutation_number_of_final_mutants'],
                                                             keff_constraint_string= "keff_cnn")

        for ind in sorted_list_of_children:
            ind.evaluated_with_cnn = True
        return sorted_list_of_children

    def apply_cnn_heuristic_constraint(self,list_of_individuals, constraint_type):
        returning_list_of_individuals = []
        for individual in list_of_individuals:
            constraint_bool_list = []
            for constraint in constraint_type:
                constraint_bool = False
                split_constraint = constraint.split("#")
                print(split_constraint)
                print(
                    "Checking {}. {}. {}. {}".format(getattr(individual, split_constraint[0]), split_constraint[2], split_constraint[0], float(split_constraint[1])))
                if split_constraint[2] == 'lesser_equal':
                    if float(getattr(individual, split_constraint[0])) < float(split_constraint[1]):
                        constraint_bool = True
                elif split_constraint[2] == 'greater':
                    if float(getattr(individual, split_constraint[0])) > float(split_constraint[1]):
                        constraint_bool = True
                else:
                    exit("CNN heuristic {} not covered".format(split_constraint[2]))
                constraint_bool_list.append(constraint_bool)
            add_individual_bool = True
            for constraint_bool_ in constraint_bool_list:
                if constraint_bool_ == False:
                    add_individual_bool = False
            if add_individual_bool:

                returning_list_of_individuals.append(individual)
        return returning_list_of_individuals



    def apply_constraint(self, list_of_individuals, constraint_type):
        print("Applying constraint:", constraint_type)
        meet_constraint = False
        ### Seperating constraint and options, etc.
        constraint_split = constraint_type.split("#")

        constraint_type_ = constraint_split[0]
        constraint_run_location = constraint_split[1]
        constraint_lin_sim_anneal = constraint_split[2]
        constraint_options = constraint_split[3]

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
                        ind.keff = random.uniform(0.5, 1.0)
                    else:
                        ind.keff = self.mcnp_file_handler.get_keff(ind.keff_input_file_string)

                    ### If that keff is above a set threshold, sets acceptable_eigenvalue to false. Else, sets it True.
                    if float(ind.keff) >= self.options['enforced_maximum_eigenvalue']:
                        print("keff,", ind.keff, "too high. Skipping source calculation")
                        ind.acceptable_eigenvalue = False
                    else:
                        ind.acceptable_eigenvalue = True

            if 'cnn' in self.options['solver']:
                ### Building list of input variables
                np_array_of_all_individual_inputs = self.build_list_of_fns_patterns_for_cnn_3x3(list_of_individuals)

                list_of_individuals = self.evaluate_cnn_objective_function(
                    'CNN_solver_objective_functions',
                    ['keff'],
                    np_array_of_all_individual_inputs,
                    list_of_individuals,
                    output_append_flag="_cnn")

                ### Grabs keff from the output files
                for ind in list_of_individuals:
                    if self.options['fake_fitness_debug']:
                        ind.keff_cnn = random.uniform(0.5, 1.0)
                    else:
                        ind.keff = ind.keff_cnn

                    ### If that keff is above a set threshold, sets acceptable_eigenvalue to false. Else, sets it True.
                    if float(ind.keff_cnn) >= self.options['enforced_maximum_cnn_eigenvalue']:
                        print("keff_cnn,", ind.keff_cnn, "too high. Skipping source calculation")
                        ind.acceptable_eigenvalue = False
                    else:
                        ind.acceptable_eigenvalue = True
                ### Building list of patterns to evaluate with CNN

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

            if constraint_lin_sim_anneal == "linearSimAnneal":

                #print(constraint_options_split)
                #print("Applying linearSimAnneal constraint")
                assert len(constraint_options_split) >= 4, "Not enough options specified in linearSimAnneal"

                _, gen_val = constraint_options_split[1].split('_')
                _, target_val = constraint_options_split[2].split('_')
                _, relaxed_constraint_Bool = constraint_options_split[3].split('_')

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

                ### If using a "relaxed" constraint that varies such that there are a minimum of number
                ### of potential parent individuals
                if relaxed_constraint_Bool == 'True':
                    number_meeting_constraint = self.check_number_of_inds_meeting_constraint(individuals=list_of_individuals,
                                                            checking_value=constraint_type_,
                                                                check_value_type=float,
                                                                check_value_float_less_or_greater='greater_equal',
                                                                check_value_target=linearSimAnneal_threshold)
                    while number_meeting_constraint < self.options['number_of_parents']:
                        #print("Reducing {} constraint. Current value: {}, Current count: {}".format(constraint_type_,
                        #                                                                            linearSimAnneal_threshold,
                        #                                                                            number_meeting_constraint))
                        linearSimAnneal_threshold -= 0.01
                        number_meeting_constraint = self.check_number_of_inds_meeting_constraint(
                            individuals=list_of_individuals,
                            checking_value=constraint_type_,
                            check_value_type=float,
                            check_value_float_less_or_greater='greater_equal',
                            check_value_target=linearSimAnneal_threshold)
                        if linearSimAnneal_threshold < 0.0:
                            print("{} individuals meet the a <0.0 constraint on {}, continuing".format(number_meeting_constraint, linearSimAnneal_threshold))
                            break


                ### Applying linearSimAnneal constraint to individuals
                for individual in list_of_individuals:
                    if getattr(individual, constraint_type_) < linearSimAnneal_threshold:
                        for objective_function in self.options['fitness']:
                            setattr(individual, objective_function, 0.0)

                        print("Setting individual:", individual.ind_count, getattr(individual, constraint_type_),constraint_type_,'to 0.0', linearSimAnneal_threshold)
                    else:
                        print("Individual Ok!", individual.input_file_string, getattr(individual, constraint_type_), 'greater than/equal to', linearSimAnneal_threshold)

        return list_of_individuals

    def non_dominated_sorting(self,
                              individuals_to_sort = 'default',
                              fitness_to_nds = 'default',
                              number_of_individuals_to_keep = 'default',
                              keff_constraint_string = 'keff'):
        front = [[]]

        if fitness_to_nds == 'default':
            fitness_to_nds = self.options['fitness']
        if individuals_to_sort == 'default':
            individuals_to_sort = self.individuals

            if self.options['sort_all_possible_individuals']:
                individuals_to_sort = self.all_individuals
        else:
            print("Length of individuals to sort: {}".format(len(individuals_to_sort)))

        ### Checking if there are any individuals that meet keff requirement
        ### If there are no individuals with an acceptable eigenvalue, sorting by decreasing keff to pressure population down
        if self.check_number_of_inds_meeting_constraint(individuals=individuals_to_sort,
                                                        checking_value = keff_constraint_string,
                                                        check_value_type = float,
                                                        check_value_target = 0.95) < self.options['number_of_parents']:
            ### Because no parents meet the criteria, sorting all individuals by keff
            ### and taking the lowest 20 to be the parents of the next generation
            print("Not enough individuals meet minimum keff requirement: {}".format(keff_constraint_string))
            individuals_to_sort.sort(key=lambda x: float(getattr(x, keff_constraint_string)), reverse=False)
            print("Finished Sorting")
            return individuals_to_sort[:self.options['number_of_parents']]



        for individual in individuals_to_sort:
            individual.front_rank = 'none'
            #print(individual.ind_count)
            #print("Nondominated sorting", individual.ind_count, individual.keff, individual.representativity)
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

                ### Checking if individual 1 dominates individual 2
                if self.check_domination(individual, individual_, fitness_to_nds):
                    if individual_ not in dominated_list:
                        dominated_list.append(individual_)

                ### Checking if individual 2 dominates individual 1
                elif self.check_domination(individual_, individual, fitness_to_nds):
                    number_of_inds_that_dominate_individual += 1




            individual.dominated_list = dominated_list
            individual.number_of_inds_that_dominate_individual = number_of_inds_that_dominate_individual
            #print("Number of inds dominated {}. Number of inds that dominate this ind: {}".format(len(individual.dominated_list), individual.number_of_inds_that_dominate_individual))
            #print(individual.input_file_string, individual.keff, individual.representativity, len(individual.dominated_list), individual.number_of_inds_that_dominate_individual)

            # If no other individuals dominate this one, it is placed on the pareto front
            if individual.number_of_inds_that_dominate_individual == 0:
                individual.front_rank = 0
                front[0].append(individual)

        ### Front counter
        pareto_front = 0

        ### Cylcing through the fronts, starting with front 0
        while front[pareto_front] != []:
            print("          pareto front:", pareto_front)
            current_front = []
            ### Visiting each dominated individual in the current pareto front
            for individual in front[pareto_front]:
                print(individual)
                for individual_ in individual.dominated_list:

                    if individual_.input_file_string == individual.input_file_string:
                        continue
                    #print(individual_.input_file_string, individual_.number_of_inds_that_dominate_individual)
                    individual_.number_of_inds_that_dominate_individual -= 1
                    #print("individual_.number_of_inds_that_dominate_individual ", individual_, individual_.number_of_inds_that_dominate_individual, individual.number_of_inds_that_dominate_individual )
                    #print(individual_.input_file_string, individual_.number_of_inds_that_dominate_individual)
                    if individual_.number_of_inds_that_dominate_individual == 0:
                        individual_.front_rank = pareto_front + 1
                        current_front.append(individual_)
                    #print("Ranks:", individual_.input_file_string, individual_.front_rank, current_front)

            front.append(current_front)
            pareto_front += 1
        #print("Pareto fronts:")
        #for front_count, front_list in enumerate(front):
        #    print(front_list)
        #    for ind in front_list:
        #        print(front_count, ind.input_file_string, ind.representativity, ind.keff)
        pareto_front = front

        #print("all inds dominated count")
        #for ind in individuals_to_sort:
            #if ind.number_of_inds_that_dominate_individual == 0:
                #continue

            #print(ind, ind.number_of_inds_that_dominate_individual, ind.dominated_list)
        #for individual_ in self.individuals:
            #print(individual_.input_file_string,  individual_.representativity, individual_.keff)
        number_of_parents = number_of_individuals_to_keep
        if number_of_individuals_to_keep == 'default':
            number_of_parents = self.options['number_of_parents']

        #print("Number of parents!", number_of_parents)
        #print("Number of fronts:", len(front))
        #for front_ in front:
        #    print("    " , str(len(front_)))
        ### Building list of parents
        parents_list = []
        for front in pareto_front:
            if front == []:
                continue
            #print("Front:", len(front))
            if len(front) < (number_of_parents - len(parents_list)):
                parents_list = parents_list + front
                print("parents_list len",len(parents_list))
            else:

                front = self.crowding_distance(front, fitness_to_nds)
                #print(len(front), self.options['number_of_parents'], len(parents_list))
                front.sort(key=lambda x: x.crowding_distance, reverse=True)

                ind_count = 0
                #print("Adding parents to parents list")
                while number_of_parents != len(parents_list):
                    parents_list.append(front[ind_count])
                    ind_count += 1
                    print("    parents list len", len(parents_list))
            #print("parent_list len, etc", len(parents_list), self.options['number_of_parents'])
        #for parent in parents_list:
            #print(parent.input_file_string, parent.keff, parent.representativity, parent.crowding_distance)
        number_of_individuals_to_keep_int = int(number_of_parents)
        if len(parents_list) < number_of_individuals_to_keep_int:
            while len(parents_list) < number_of_individuals_to_keep_int:
                dominated_by_count = 1
                for ind in individuals_to_sort:
                    if len(parents_list) < number_of_individuals_to_keep_int:
                        if ind.number_of_inds_that_dominate_individual == dominated_by_count:
                            parents_list.append(ind)
                    else:
                        break

                dominated_by_count += 1
        print(len(parents_list))

        return parents_list

    def crowding_distance(self, front, fitness_list):


        if front == []:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for fitness in fitness_list:
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
    def check_domination(self, ind_1, ind_2, fitness_options):

        for fit_count, fitness_ in enumerate(fitness_options):
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
                                print("Evaluating representativity!!!", individual.ran_source_calculation)
                                ### Building and submitting MCNP input file
                                self.build_and_run_mcnp_evaluation_input(individual)
                                individual.ran_source_calculation = True
                                print(individual.ran_source_calculation)
                    self.wait_on_jobs('mcnp')

                if 'cnn' in self.options['solver']:
                    ### Building list of input variables
                    np_array_of_all_individual_inputs = self.build_list_of_fns_patterns_for_cnn_3x3(list_of_individuals)

                    list_of_individuals = self.evaluate_cnn_objective_function(
                        'CNN_solver_objective_functions',
                        ['representativity'],
                        np_array_of_all_individual_inputs,
                        list_of_individuals,
                        output_append_flag="")

                    for individual in list_of_individuals:
                        if individual.acceptable_eigenvalue == False:
                            individual.representativity = 0.0
            if evaluation_type == 'total_flux':
                print("Evaluating Total Flux")
                self.mcnp_inputs = []
                if 'mcnp' in self.options['solver']:
                    for individual in list_of_individuals:
                        if individual.ran_source_calculation == False:
                            if individual.acceptable_eigenvalue == True:
                                print("Evaluating total flux!!!", individual.ran_source_calculation)
                                ### Building and submitting MCNP input file
                                self.build_and_run_mcnp_evaluation_input(individual)
                                individual.ran_source_calculation = True
                    self.wait_on_jobs('mcnp')

                if 'cnn' in self.options['solver']:
                    ### Building list of input variables
                    np_array_of_all_individual_inputs = self.build_list_of_fns_patterns_for_cnn_3x3(list_of_individuals)

                    list_of_individuals = self.evaluate_cnn_objective_function(
                        'CNN_solver_objective_functions',
                        ['total_flux'],
                        np_array_of_all_individual_inputs,
                        list_of_individuals,
                        output_append_flag="")

                    ### Setting total_flux to total_flux_cnn
                    for individual in list_of_individuals:
                        if individual.acceptable_eigenvalue == False:
                            individual.total_flux = 0.0
                        else:
                            individual.total_flux = individual.total_flux / 4e9
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

                if 'cnn' in self.options['solver']:
                    ### Building list of input variables
                    np_array_of_all_individual_inputs = self.build_list_of_fns_patterns_for_cnn_3x3(list_of_individuals)

                    list_of_individuals = self.evaluate_cnn_objective_function(
                        'CNN_solver_objective_functions',
                        ['int_exp'],
                        np_array_of_all_individual_inputs,
                        list_of_individuals,
                        output_append_flag="")

                    ### Setting total_flux to total_flux_cnn
                    for ind_ in list_of_individuals:
                        ind_.integral_keff = ind_.int_exp + ind_.keff
                        ind_.integral_keff_value = ind_.integral_keff
                        if ind_.acceptable_eigenvalue == False:
                            ind_.integral_keff_value = 0.0
                            ind_.integral_keff = 0.0

                        ### Pulling evaluation data
        if 'mcnp' in self.options['solver']:
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
                                    individual.flux_values, lethargy_adjustment = self.options['Lethargy_Adjustment_of_Rep'], bins = self.options['energy_bins'])
                            if individual.acceptable_eigenvalue == False:
                                individual.representativity = 0.0

                        if evaluation_type == 'total_flux':
                            if individual.acceptable_eigenvalue == True:
                                individual.flux_values, individual.flux_uncertainty, individual.total_flux, individual.total_flux_unc = self.mcnp_file_handler.get_f4_flux_from_output(
                                    individual.input_file_string + "o")
                                individual.representativity = self.mcnp_file_handler.calculate_representativity_v3(
                                    individual.flux_values,lethargy_adjustment = self.options['Lethargy_Adjustment_of_Rep'], bins = self.options['energy_bins'])
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

    def check_number_of_inds_meeting_constraint(self,
                                                individuals = [],
                                                checking_value = 'acceptable_eigenvalue',
                                                check_value_type = 'default',
                                                check_value_float_less_or_greater = 'lesser_equal',
                                                check_value_target = 'default',
                                                stop_on_count = 'parent_count',
                                                stop_on_count_bool = False):
        ### Setting stop on count value
        if stop_on_count == 'parent_count':
            stop_count_value = self.options['number_of_parents']
        else:
            stop_count_value = int(stop_on_count)

        ### Getting individuals to do check on
        if type(individuals) == str:
            individuals_to_check = getattr(self, individuals)
        if type(individuals) == list:
            assert individuals != [], "Individuals not specified for constraint check"
            individuals_to_check = individuals


        ### Checking to see if any individuals have met constraint
        number_of_acceptable_individuals = 0
        for ind in individuals_to_check:
            #assert check_value_type == 'default', "Constraint type not specified for constraint {}".format(checking_value)

            if (check_value_type == True) or (check_value_type == False):
                if getattr(ind, checking_value) == check_value_type:
                    number_of_acceptable_individuals += 1

            if check_value_type == float:
                assert check_value_target != 'default', "Float check value not specified for constraint {}".format(checking_value)
                #print("Checking value {} against target value {}. {} {}".format(getattr(ind, checking_value),check_value_target,checking_value, check_value_float_less_or_greater))
                ### Checking various ways which constraint could be applied
                if check_value_float_less_or_greater == 'lesser_equal':
                    if float(getattr(ind, checking_value)) <= check_value_target:
                        number_of_acceptable_individuals += 1
                if check_value_float_less_or_greater == 'greater_equal':
                    if float(getattr(ind, checking_value)) >= check_value_target:
                        number_of_acceptable_individuals += 1
                if check_value_float_less_or_greater == 'equal':
                    if float(getattr(ind, checking_value)) == check_value_target:
                        number_of_acceptable_individuals += 1
                if check_value_float_less_or_greater == 'lesser':
                    if float(getattr(ind, checking_value)) < check_value_target:
                        number_of_acceptable_individuals += 1
                if check_value_float_less_or_greater == 'greater':
                    if float(getattr(ind, checking_value)) > check_value_target:
                        number_of_acceptable_individuals += 1

            if stop_on_count_bool:
                if number_of_acceptable_individuals == stop_count_value:
                    print("There are enough individuals meeting {} constraint to make complete parent set, continuing".format(number_of_acceptable_individuals))
                    return number_of_acceptable_individuals

        print(number_of_acceptable_individuals, "meet", checking_value, "with target of",check_value_target)
        return number_of_acceptable_individuals

    def predict_keff_with_cnn(self, new_child_ind):
        material_cnn_form_preproccessed = self.cnn_handler.preprocess_variable_cassette_child_object(
            new_child_ind)
        # print("material_cnn_form_preproccessed", material_cnn_form_preproccessed)
        material_cnn_form, _ = self.cnn_handler.build_3x3_fns_data_variable_cassette_A(single_case=True,
                                                                                       single_case_material_matrix=material_cnn_form_preproccessed)
        ### Predicting keff
        pred_keff = self.cnn_handler.model.predict(material_cnn_form)[0][0]
        return pred_keff

    ### The crossover function creates total population - number of parents
    def crossover(self, keff_constraint_string = 'keff'):

        if self.options['solver'] == 'cnn':
            self.cnn_handler.load_cnn_model(self.options['CNN_solver_objective_functions']['keff'])

        if self.options['use_non_dominated_sorting']:
            self.parents_list = self.non_dominated_sorting(keff_constraint_string = keff_constraint_string)
            print("self.check_number_of_inds_meeting_constraint(individuals = self.parents_list)", self.check_number_of_inds_meeting_constraint(individuals = self.parents_list))

        else:
            if self.check_number_of_inds_meeting_constraint(individuals = self.parents_list) > 0:
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
                valid_child = False
                ### Count of tried children
                child_try_count = 1
                best_keff_so_far = 10.0
                while valid_child == False:


                ### Creating child individual
                    child_ind = individual.individual(self.options, self.generation, self.individual_count)
                    self.individual_count += 1

                    new_child_ind = self.bitwise_crossover(parent_1, parent_2, child_ind, material_matrix_value = "material_matrix")

                    if self.options['adjustable_zone_2A_cassette_bool']:
                        new_child_ind = self.bitwise_crossover(parent_1, parent_2, new_child_ind, material_matrix_value = "material_matrix_cassette_2A")

                    ### Evaluating child with CNN, if it guesses that the child is below threshold then it is valid
                    if self.options['CNN_check_keff_crossover']:
                        ### Putting material matrix into form expected by CNN
                        pred_keff = self.predict_keff_with_cnn(new_child_ind)

                        new_child_ind.cnn_keff = pred_keff
                        new_child_ind.evaluated_with_cnn = True
                        #print("pred_keff \t{}\t{}".format(pred_keff, new_child_ind.cnn_keff))

                        if pred_keff < best_keff_so_far:
                            best_child = copy.deepcopy(new_child_ind)
                            print("best keff and new best \t{}\t{}".format(best_keff_so_far, pred_keff))
                            best_keff_so_far = copy.deepcopy(pred_keff)

                        if pred_keff < self.options['CNN_check_keff_threshold']:
                            print("Child is valid", new_child_ind.cnn_keff)
                            valid_child = True

                        if child_try_count == self.options['CNN_check_keff_number_of_times_to_try_to_create_valid_child']:

                            new_child_ind = copy.deepcopy(best_child)
                            valid_child = True
                            new_child_ind.keff_cnn = best_keff_so_far



                        #print('child_try_count', child_try_count)
                        #print("Final best child keff:", new_child_ind.cnn_keff)
                        if valid_child == False:
                            self.individual_count -= 1

                    else:
                        valid_child = True
                    child_try_count += 1
                        ### If keff is below threshold, setting valid_child to True, breaking the loop

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

        #print("parent 1 pattern:", material_matrix_value, getattr(parent_1, material_matrix_value), len(getattr(parent_1, material_matrix_value)))
        #print("parent 2 pattern:", material_matrix_value, getattr(parent_2, material_matrix_value), len(getattr(parent_2, material_matrix_value)))
        #print("Child pattern before:", getattr(child_ind, material_matrix_value), child_ind.ind_count)

        ### Choosing either parent's length
        parent_selection = random.randint(0, 1)

        material_matrix_ = getattr(parent_1, material_matrix_value)
        if parent_selection == 1:
            material_matrix_ = getattr(parent_2, material_matrix_value)

        temp_material_master_list = []
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
                        material = getattr(parent_2, material_matrix_value)[material_list_count][material_count]
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

        #print("Child pattern after:", material_matrix_value, getattr(child_ind, material_matrix_value), len(getattr(child_ind, material_matrix_value)), child_ind.ind_count)
        return child_ind

    def singlepoint_crossover(self, parent_1, parent_2):
        child_ind = individual.individual(self.options, self.generation, self.individual_count)
        self.individual_count += 1
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
                            random_new_material = random.randint(0, len(self.options['adjustable_zone_2A_material_types']) - 1)
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
        returned_list_of_mutated_individuals = []
        ### Currently only works on a material-basis
        if mutating_debug:
            print("MUTATING!!!")
        if self.options['solver'] == 'cnn':
            self.cnn_handler.load_cnn_model(self.options['CNN_solver_objective_functions']['keff'])

        if self.options['mutation_type'] == 'bitwise':
            if mutating_debug:
                print("BITWISE!", len(list_of_individuals))
            for ind_count, individual in enumerate(list_of_individuals):
                if mutating_debug:
                    print("MUTATING:", ind_count)

                ### Preserving an original copy of the material matrix
                original_material_matrix = copy.deepcopy(getattr(individual, material_matrix_value))




                if self.options['CNN_check_keff_mutation']:
                    print("\t\t\tCNN CHECK MUTATION!")
                    valid_child = False
                    ### Count of tried children
                    child_try_count = 1
                    best_keff_so_far = 10.0
                    while valid_child == False:


                    ### Creating child individual
                        #child_ind = individual.individual(self.options, self.generation, self.individual_count)
                        self.individual_count += 1

                        ### Doing single bit mutation
                        setattr(individual, material_matrix_value, self.single_bit_mutation(original_material_matrix, mutating_debug = mutating_debug))

                        ### If the cassette is a variable length, doing that mutation
                        if do_variable_size_mutation:

                            setattr(individual, material_matrix_value, self.variable_cassette_length_mutation(original_material_matrix, mutating_debug = mutating_debug))

                            ### Resetting the number of plates in this individual to the correct value
                            setattr(individual, "number_of_plates_in_cassette_2A", len(getattr(individual, material_matrix_value)))

                        ### Evaluating child with CNN, if it guesses that the child is below threshold then it is valid

                        ### Putting material matrix into form expected by CNN
                        pred_keff = self.predict_keff_with_cnn(individual)

                        individual.cnn_keff = pred_keff
                        individual.evaluated_with_cnn = True
                        print("pred_keff \t{}\t{}".format(pred_keff, individual.cnn_keff))

                        if pred_keff < best_keff_so_far:
                            best_child = copy.deepcopy(individual)
                            print("best keff and new best \t{}\t{}".format(best_keff_so_far, pred_keff))
                            best_keff_so_far = copy.deepcopy(pred_keff)

                        if pred_keff < self.options['CNN_check_keff_threshold']:
                            print("Child is valid", individual.cnn_keff)
                            valid_child = True

                        if child_try_count == self.options['CNN_check_keff_number_of_times_to_try_to_create_valid_child']:

                            individual = copy.deepcopy(best_child)
                            valid_child = True
                            individual.keff_cnn = best_keff_so_far



                        #print('child_try_count', child_try_count)
                        #print("Final best child keff:", individual.cnn_keff)
                        if valid_child == False:
                            self.individual_count -= 1
                        else:
                            valid_child = True
                        child_try_count += 1
                            ### If keff is below threshold, setting valid_child to True, breaking the loop
                else:
                ### Doing single bit mutation
                    setattr(individual, material_matrix_value, self.single_bit_mutation(original_material_matrix, mutating_debug = mutating_debug))

                    ### If the cassette is a variable length, doing that mutation
                    if do_variable_size_mutation:

                        setattr(individual, material_matrix_value, self.variable_cassette_length_mutation(original_material_matrix, mutating_debug = mutating_debug))

                        ### Resetting the number of plates in this individual to the correct value
                        setattr(individual, "number_of_plates_in_cassette_2A", len(getattr(individual, material_matrix_value)))



                if self.options['verify_fuel_mass_after_mutation']:
                    ### Checking if new child meets fuel # requirement
                    fuel_count = individual.count_material(self.options['fuel_material_number'])
                    while ((fuel_count > self.options['maximum_fuel_elements']) or (
                            fuel_count < self.options['minimum_fuel_elements'])):
                        setattr(individual, material_matrix_value, self.single_bit_mutation(original_material_matrix))
                        fuel_count = individual.count_material(self.options['fuel_material_number'])

            returned_list_of_mutated_individuals.append(individual)
        return list_of_individuals

    def single_bit_mutation(self,
                            material_matrix,
                            force_mutation = False,
                            force_mutation_per_material_sublist = 1,
                            mutating_debug = False):
        if mutating_debug:
            print("Single bit mutation!")
            print("Material matrix before mutation:")
            print(material_matrix)
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

        for material_count_mm, material_list in enumerate(material_matrix):
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
                    if mutating_debug:
                        print("Mutating material count: {} {} material before: {}".format(material_count_mm, material_count, material))

                    new_index = random.randint(0, len(self.options['material_types']) - 1)
                    while material == self.options['material_types'][new_index]:
                        new_index = random.randint(0, len(self.options['material_types']) - 1)
                    # print("NEW_INDEX: ", new_index, len(self.options['material_types']) - 1)
                    # print("new material: ", self.options['material_types'][new_index], "old", material)
                    material = self.options['material_types'][new_index]
                    if mutating_debug:
                        print("New material: {}".format(material))
                material_matrix_sublist.append(material)
            new_material_matrix.append(material_matrix_sublist)
        #print("new_material_matrix:", new_material_matrix)
        if mutating_debug:
            print("New material matrix: {}".format(new_material_matrix))
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
            ### Evaluating with cnn if needed
            if self.options['CNN_check_keff_mutation'] or self.options['CNN_check_keff_crossover']:
                individual.keff_cnn = self.predict_keff_with_cnn(individual)

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
            if write_option == 'CNN_check_keff_threshold':
                write_string += str(self.options['CNN_check_keff_threshold']) + ","
            if write_option == 'keff_cnn':
                write_string += str(individual.keff_cnn) + ","
            if write_option == 'representativity_cnn':
                write_string += str(individual.representativity_cnn) + ","
            if write_option == 'total_flux_cnn':
                write_string += str(individual.total_flux_cnn) + ","

            if write_option == 'evaluated_with_cnn':
                write_string += str(individual.evaluated_with_cnn) + ","

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
