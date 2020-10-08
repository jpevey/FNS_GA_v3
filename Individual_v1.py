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

        self.grid_x = options['grid_x']
        self.grid_y = options['grid_y']
        self.options = options
        self.ind_count = individual_count
        self.total_flux = 0.0
        self.input_file_string = self.options['file_keyword']+ "_gen_" + str(generation) + "_ind_" + str(individual_count) + ".inp"
        self.keff_input_file_string = "keff_" + self.input_file_string
        self.create_random_pattern()
        self.parent_string = "random_initialized,N/A,"
        self.born_from_crossover = False
        self.ran_source_calculation = False

        self.acceptable_eigenvalue = True

        self.default_materials = collections.OrderedDict()

    def create_random_pattern(self):
        #print("Creating random pattern for ind:", self.ind_count)
        self.pattern = collections.OrderedDict()
        number_of_fuel = random.randint(self.options['minimum_fuel_elements'], self.options['maximum_fuel_elements'])
        self.fuel_locations = []
        total_number_of_locations = self.grid_y * self.grid_x
        #print("NUMBER OF FUEL: " , number_of_fuel, self.ind_count)
        ### Assigning fuel locations
        for _ in range(number_of_fuel):
            fuel_location = random.randint(1, total_number_of_locations)
            while fuel_location in self.fuel_locations:
                fuel_location = random.randint(1, total_number_of_locations )
            self.fuel_locations.append(fuel_location)

        #print("fuel locations:", self.fuel_locations)

        self.create_material_matrix()

    def create_material_matrix(self):
        ### Creating list of materials in this ind.
        material_matrix = []
        material_count = 0
        for _ in range(self.grid_x):
            minor_material = []
            for __ in range(self.grid_y):
                material_count += 1
                material_index = random.randint(1, len(self.options['material_types']))
                material = self.options['material_types'][material_index - 1]
                if material_count in self.fuel_locations:
                    material = 1
                minor_material.append(material)
            material_matrix.append(minor_material)
        self.material_matrix = material_matrix
        print("Material Matrix:", self.material_matrix)

    def find_fuel_location(self):
        for count, mat in enumerate(self.material_matrix):
            #print(count, mat[0])
            if mat[0] == self.options['fuel_index']:
                #print("Found fuel in location index: ", count, 2.54 / 4 + count * self.options['fuel_index_multiplier'])
                return 2.54 / 4 + count * self.options['fuel_index_multiplier']

    def create_discrete_material_mcnp_dictionary(self, keywords_list = []):
        if keywords_list == []:
            keywords_list = self.options['keywords_list']

        material_dictionary = collections.OrderedDict()
        for count, item in enumerate(keywords_list):
            #print('self.material_matrix[count]', self.material_matrix[count][0], self.options['default_mcnp_mat_count_and_density'][self.material_matrix[count][0]])
            material_dictionary[item] = self.options['default_mcnp_mat_count_and_density'][self.material_matrix[count][0]]
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

    def make_material_string_csv(self):
        ### Making a string for the input file
        material_string = ""
        for row_materials in self.material_matrix:

            for material in row_materials:
                material_string = material_string + str(material) + ","
        # print("Making csv string:", self.ind_count, material_string.strip())
        self.material_string_csv = material_string.strip()
        return self.material_string_csv

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

