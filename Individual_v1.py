import collections
import random

random.seed(865)


class individual:
    def __init__(self, options, generation, individual_count):
        print("Created Individual", individual_count)
        self.evaluated = False
        self.keyword_strings = collections.OrderedDict()
        self.keff = 0.0
        self.generation = generation
        self.options = options
        self.ind_count = individual_count
        self.total_flux = 0.0
        self.input_file_string = self.options['file_keyword'] + "_gen_" + str(generation) + "_ind_" + str(individual_count) + ".inp"
        self.keff_input_file_string = "keff_" + "gen_" + str(generation) + "_ind_" + str(individual_count) + ".inp"

        ### building intiial pattern for materials
        self.create_random_pattern(self.options['enforce_material_count_on_creation'],
                                   grid_x_value = self.options['grid_x'],
                                   grid_y_value = self.options['grid_y'])

        ### if using the variable cassette 2 option, building that pattern
        if options['adjustable_zone_2A_cassette_bool']:
            self.apply_special_operator(options['adjustable_zone_2A_individual_attributes'],
                                        options['variable_cassette_2A_debug'])

            ### Building the list of keywords for the adjustable cassette. It defaults to build
            ### enough keywords for all possible plates, which are "" if beyond the length of the
            ### cassette
            self.options['adjustable_zone_2A_keywords_list'] = []
            for x in range(self.options['adjustable_zone_2A_fixed_values']['maximum_plates']):
                self.options['adjustable_zone_2A_keywords_list'].append(
                    self.options['cassette_2A_keywords_template'] + str(x + 1) +"_")


            self.create_random_pattern(enforce_material_bool = False,
                                  material_locations_val = "material_locations_cassette_2A",
                                  material_matrix_setattr_val = "material_matrix_cassette_2A",
                                  total_number_of_locations = self.number_of_plates_in_cassette_2A,
                                  material_types = 'adjustable_zone_2A_material_types',
                                  grid_x_value = self.number_of_plates_in_cassette_2A,
                                  grid_y_value = 1)

        self.parent_string = "random_initialized,N/A,"
        self.born_from_crossover = False
        self.ran_source_calculation = False

        self.acceptable_eigenvalue = True

        self.default_materials = collections.OrderedDict()

    def apply_special_operator(self, operator_dict, debug=False):
        if debug:
            print("operator_dict", operator_dict)

        for val in operator_dict:
            if debug:
                print(val)

            val_options = operator_dict[val]
            init_type, init_options = val_options['init'].split(",")
            init_options = init_options.split("%")
            if debug:
                print("init_type, init_options", init_type, init_options)

            assert init_type == 'random', "Special operator {} not handled. Only 'random' works right now...".format(
                init_type)

            if init_type == 'random':
                if init_options[0] == 'int':

                    if init_options[1] in self.options['adjustable_zone_2A_fixed_values']:
                        minimum_value = self.options['adjustable_zone_2A_fixed_values'][init_options[1]]
                    else:
                        minimum_value = init_options[1]

                    if init_options[2] in self.options['adjustable_zone_2A_fixed_values']:
                        maximum_value = self.options['adjustable_zone_2A_fixed_values'][init_options[2]]
                    else:
                        maximum_value = init_options[2]

                    init_value = random.randint(int(minimum_value), int(maximum_value))

                    if debug:
                        print("minimum value {}, maximum value {}, init value {}".format(minimum_value, maximum_value,
                                                                                         init_value))

            setattr(self, val, init_value)

    def create_random_pattern(self, enforce_material_bool,
                              grid_x_value,
                              grid_y_value,
                              material_locations_val = "material_locations",
                              material_matrix_setattr_val = "material_matrix",
                              minimum_material_elements = "minimum_material_elements",
                              maximum_material_elements = "maximum_material_elements",
                              total_number_of_locations = "default",
                              material_types = 'material_types',
                              option_enforce_material_number = 'enforce_material_number'):

        #self.options['enforce_material_count_on_creation']
        #print("Creating random pattern for ind:", self.ind_count)
        if enforce_material_bool == True:
            ### Setting minimum and maximum values for specified material
            minimum_number = self.options[minimum_material_elements]
            maximum_number = self.options[maximum_material_elements]

            ### The default behavior. Modifying it to account for variable cassette 2A function
            if total_number_of_locations == "default":
                total_number_of_locations = self.options["grid_y"] * self.options["grid_x"]


            number_of_material = random.randint(minimum_number, maximum_number)

            setattr(self, material_locations_val, [])

            for _ in range(number_of_material):
                ### Pick a location for the material that the limit is being imposed on
                material_location = random.randint(1, total_number_of_locations)
                ### If the value is already in the list "material locations val" it picks another
                while material_location in getattr(self, material_locations_val):
                    material_location = random.randint(1, total_number_of_locations)
                ### Append the value to the list
                getattr(self, material_locations_val).append(material_location)

        ### Creating the material matrix
        self.create_material_matrix(material_matrix_setattr_val = material_matrix_setattr_val,
                                    option_grid_x_val = grid_x_value,
                                    option_grid_y_val = grid_y_value,
                                    option_material_types = self.options[material_types],
                                    option_enforce_material_bool = enforce_material_bool ,
                                    option_enforce_material_number = self.options[option_enforce_material_number])

    def debug_fake_fitness(self, fitness_type, keff_is_acceptable):
        if keff_is_acceptable:
            setattr(self, fitness_type, random.uniform(0, 1.0))
        else:
            setattr(self, fitness_type, 0.0)
        return

    def create_material_matrix(self,
                               material_matrix_setattr_val,
                               option_grid_x_val,
                               option_grid_y_val,
                               option_material_types,
                               option_enforce_material_bool,
                               option_enforce_material_number):

        ### Creating list of materials in this ind.
        material_matrix = []
        material_count = 0

        material_types = option_material_types
        if option_enforce_material_bool:
            enforce_material_number = option_enforce_material_number

        for _ in range(option_grid_x_val):
            minor_material = []
            for __ in range(option_grid_y_val):
                material_count += 1
                material_index = random.randint(1, len(material_types))
                material = material_types[material_index - 1]
                if option_enforce_material_bool:
                    if material_count in self.material_locations:
                        material = enforce_material_number
                    else:
                        material_index = random.randint(1, len(material_types))
                        while material_index == enforce_material_number:
                            material_index = random.randint(1, len(material_types))
                        material = material_types[material_index - 1]
                minor_material.append(material)
            material_matrix.append(minor_material)
        setattr(self, material_matrix_setattr_val, material_matrix)

    def find_fuel_location(self):
        for count, mat in enumerate(self.material_matrix):
            #print(count, mat[0])
            if mat[0] == self.options['fuel_index']:
                #print("Found fuel in location index: ", count, 2.54 / 4 + count * self.options['fuel_index_multiplier'])
                return 2.54 / 4 + count * self.options['fuel_index_multiplier']

    def build_variable_cassette_2a_dictionary(self, dictionary_):
        fixed_value_dict = self.options['adjustable_zone_2A_fixed_values']
        '''options['adjustable_zone_2A_fixed_values'] = {'maximum_plates': 30,
                                                      'minimum_plates': 1,
                                                      'maximum_cassette_2A_value_cm': 52.717,
                                                      'cassette_2A_wall_thickness_cm': 0.3175,
                                                      'cassette_2A_plate_thickness_cm': 1.27,
                                                      'cassette_2A_rpp_origin': -1.27,
                                                      'depth_of_exp_volume_cm': 6*2.54}'''
        dictionary_['cassette_pattern_2A_fill_value'] = str("fill=0:"+str(int(self.number_of_plates_in_cassette_2A - 1)))
        dictionary_['cassette_pattern_2A_trcl_value'] = fixed_value_dict['maximum_cassette_2A_value_cm']\
                                                                   + 2 * fixed_value_dict['cassette_2A_wall_thickness_cm']\
                                                                   + fixed_value_dict['cassette_2A_rpp_origin'] \
                                                                   - self.number_of_plates_in_cassette_2A\
                                                                    * fixed_value_dict['cassette_2A_plate_thickness_cm']
        dictionary_['cassette_pattern_2A_cassette_inner_length'] = self.number_of_plates_in_cassette_2A\
                                                                    * fixed_value_dict['cassette_2A_plate_thickness_cm']\
                                                                    + fixed_value_dict['cassette_2A_rpp_origin']
        dictionary_['cassette_pattern_2A_cassette_outer_length'] = dictionary_['cassette_pattern_2A_cassette_inner_length']\
                                                                            + fixed_value_dict['cassette_2A_wall_thickness_cm']

        dictionary_['cassette_pattern_2A_exp_vol_max'] = dictionary_['cassette_pattern_2A_trcl_value']\
                                                         - fixed_value_dict['cassette_2A_plate_thickness_cm']\
                                                         - fixed_value_dict['cassette_2A_wall_thickness_cm']
        dictionary_['cassette_pattern_2A_exp_vol_min'] = dictionary_['cassette_pattern_2A_exp_vol_max']\
                                                         - fixed_value_dict['depth_of_exp_volume_cm']

        return dictionary_

    def create_discrete_material_mcnp_dictionary(self,material_matrix_val, keywords_list, data_dict = ""):
        ### Creating the dictionary if not specified
        if data_dict == "":
            material_dictionary = collections.OrderedDict()
        else:
            material_dictionary = data_dict

        ### Putting in blanks as default values (for variable cassette 2a)
        for count, item in enumerate(keywords_list):
            material_dictionary[item] = ""

        ### Filling data dict with correct values based on material matrix
        for count, item in enumerate(keywords_list):
            ### Adding the correct material value that MCNP expects to the dictionary
            ### If there's not a value in material_matrix_val, continuing
            try:
                material_dictionary[item] = self.options['default_mcnp_mat_count_and_density'][getattr(self, material_matrix_val)[count][0]]
            except:
                material_dictionary[item] = ""
        return material_dictionary

    def setup_scale(self, generation):
        if self.options['skip_writing_files'] == True:
            return
        self.create_scale_input(generation)
        self.create_scale_script()
        return self.script_filename

    def create_scale_input(self, generation):

        ### Creating file
        scale_input_filename = self.input_file_string
        output_file = open(self.input_file_string, 'w')
        self.scale_input_filename = scale_input_filename

        ### Printing debug
        # print("Creating scale debug::::", self.ind_count)
        # for key in self.keyword_strings:
        #    print(key, self.keyword_strings[key])

        with open(self.options['scale_template_file_string']) as template_file:
            for line in template_file:
                for keyword in self.options['template_keywords']:
                    if keyword in line:
                        line = line.replace(keyword, self.keyword_strings[keyword])
                output_file.write(line)

    def create_scale_script(self):
        if self.options['skip_writing_files'] == True:
            return
        script_filename = self.scale_input_filename + ".sh"
        self.script_filename = script_filename
        script_file = open(script_filename, 'w')

        template = self.options['scale_script_template']

        script_file.write(template.replace("%%%input%%%", self.scale_input_filename))
        script_file.close()

    def get_scale_keff(self):
        # print("self.script_filename", self.script_filename)
        split_1 = self.script_filename.split(".inp")
        scale_output_filename = split_1[0] + ".out"
        try:
            for line in open(scale_output_filename, 'r'):

                if "best estimate system k-eff" in line:
                    line_split = line.split("best estimate system k-eff")
                    line_split_2 = line_split[1].split("+")
                    keff = line_split_2[0].strip()
                    self.keff = float(keff)
        except:
            print("Couldn't get keff...")

    def make_material_string_scale(self, keyword):
        print("CREATING MATERIAL STRING FOR IND::", self.ind_count)
        if self.options['geometry'] == 'cyl':
            ### Making a string for the input file
            material_string = ""
            # print(self.material_matrix)

            cyl_count = 0
            for row_materials in self.material_matrix:

                for material in row_materials:
                    cyl_string_1 = self.options['cyl_scale_template'][cyl_count]
                    cyl_string_2 = cyl_string_1.replace('material', str(material))
                    material_string = material_string + cyl_string_2

                    cyl_count += 1

                material_string += "\n"
            self.keyword_strings[keyword] = material_string.strip()
            # print(self.keyword_strings[keyword])

        if self.options['geometry'] == 'grid':
            ### Making a string for the input file
            material_string = ""
            #print(self.material_matrix)
            for row_materials in self.material_matrix:
                for material in row_materials:
                    material_string = material_string + str(material) + " "
                material_string += "\n"
            self.keyword_strings[keyword] = material_string.strip()

    def make_material_string_csv(self, material_matrix_value, target_string_val):
        ### Making a string for the input file
        material_string = ""
        material_matrix = getattr(self, material_matrix_value)
        for row_materials in material_matrix:
            for material in row_materials:
                material_string = material_string + str(material) + ","
        # print("Making csv string:", self.ind_count, material_string.strip())
        setattr(self, target_string_val, material_string.strip())
        return material_string.strip()

    def count_material(self, material_to_count):
        mat_count = 0
        for _ in self.material_matrix:
            for material in _:
                if str(material) == str(material_to_count):
                    mat_count += 1
        return mat_count

    def enforce_material_count(self, material_value, material_count_to_enforce):
        ### Getting list of material in question locations
        material_location_list = []
        for mat_count, material_ in enumerate(self.options['material_types']):
            material_location_list.append([])
            for list_count, _ in enumerate(self.material_matrix):
                for material_count, material in enumerate(_):
                    if material == material_:
                        material_location_list[mat_count].append([list_count, material_count])

        ### Finding target material
        for _count, _ in enumerate(material_location_list):
            # print("Found", len(_), "of material:", options['material_types'][_count])
            if self.options['material_types'][_count] == material_value:
                adjustment = material_count_to_enforce - len(_)
                material_index = _count
                # print("Adjusting", material_value, "to #", material_count_to_enforce, "by", adjustment)

        ### Removing Material Elements
        while adjustment < 0:
            ### Select material to replace the target with
            replacement_int = random.randint(0, len(self.options['material_types']) - 1)

            ### Checks if the new material is different than the target material_value
            while self.options['material_types'][replacement_int] == material_value:
                replacement_int = random.randint(0, len(self.options['material_types']) - 1)

            ### Select material location to turn into replacement
            material_location_index = random.randint(0, len(material_location_list[material_index]) - 1)

            ### Switching material to new material
            material_list_index_0 = material_location_list[material_index][material_location_index][0]
            material_list_index_1 = material_location_list[material_index][material_location_index][1]

            self.material_matrix[material_list_index_0][material_list_index_1] = self.options['material_types'][
                replacement_int]

            adjustment = material_count_to_enforce - self.count_material(material_value)

        ### Adding material element(s)
        while adjustment > 0:
            ### Select material to replace the target with
            target_int = random.randint(0, len(self.options['material_types']) - 1)
            ### Checks if the new material is different than the target material_value
            while self.options['material_types'][target_int] == material_value:
                target_int = random.randint(0, len(self.options['material_types']) - 1)

            ### Select material location to turn into replacement
            material_location_index = random.randint(0, len(material_location_list[target_int]) - 1)

            ### Switching material to new material
            material_list_index_0 = material_location_list[target_int][material_location_index][0]
            material_list_index_1 = material_location_list[target_int][material_location_index][1]

            self.material_matrix[material_list_index_0][material_list_index_1] = self.options['material_types'][
                material_index]

            adjustment = material_count_to_enforce - self.count_material(material_value)

    def get_material_locations(self, material_value):
        index_value = 0
        for list_count, _ in enumerate(self.material_matrix):
            for material_count, material in enumerate(_):
                index_value += 1

    def evaluate_eigenvalue(self):
        if self.options['solver'] == 'mcnp':
            print("Evaluating with mcnp")




    def get_eigenvalue(self):
        pass

