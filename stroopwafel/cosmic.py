import os
from .utils import *
import shutil
from scipy.stats import multivariate_normal, entropy
from .distributions import Gaussian
from .classes import Location
from .constants import *
import sys
# Add pandas import
import pandas as pd
import time
from schwimmbad import MultiPool

class Cosmic:

    def __init__(self, total_num_systems, num_batches_in_parallel, num_samples_per_batch, output_folder, output_filename, cosmic_filename, debug = False, run_on_helios = True, mc_only = False):
        self.total_num_systems = total_num_systems
        # self.num_batches_in_parallel = num_batches_in_parallel #for COMPAS
        self.num_batches_in_parallel = 1 #for cosmic
        self.num_samples_per_batch = num_samples_per_batch
        self.output_folder = output_folder
        self.output_filename = os.path.join(self.output_folder, output_filename)
        self.cosmic_filename = os.path.join(self.output_folder, cosmic_filename)
        self.debug = debug
        self.run_on_helios = run_on_helios
        self.mc_only = mc_only
        # Initialize lists to collect bpp and initC DataFrames from batches
        self.all_bpp = []
        self.all_initC = []
        # Initialize bin_num counter
        self.current_bin_num = 0
        self.evolve_time = 0
        self.reject_time = 0

    def update_fraction_explored(self):
        """
        Function which updates the fraction of region which is already explored
        """
        unidentified_region_weight = 1.0 / (self.fraction_explored * self.total_num_systems)
        target_rate = float(self.num_hits) / self.num_explored
        numerator = target_rate * (np.sqrt(1. - target_rate) - np.sqrt(unidentified_region_weight))
        denominator = np.sqrt(1. - target_rate) * (np.sqrt(unidentified_region_weight * (1. - target_rate)) + target_rate)
        self.fraction_explored = 1 - numerator / denominator

    def should_continue_exploring(self):
        """
        Function that estimates if we should continue exploring or are we ready
        OUT:
            bool : boolean value telling If we should continue exploring or not
        """
        if self.mc_only:
            return self.num_explored < self.total_num_systems
        return self.num_explored / self.total_num_systems < self.fraction_explored

    def initialize(self, dimensions, interesting_systems_method, configure_code_run, rejected_systems_method, update_properties_method = None):
        """
        This function is the one which is run only once in the stroopwafel class. It initializes the associated variables and the function calls that user will specify
        IN:
            interesting_system_method: The method provided by the user which will determine what an interesting system is for stroopwafel
            update_properties_method: The method provided by the user which will run to update the properties of each of the location once it is sampled
            configure_code_run: The method provided by the user which will be running for each of the batches to determine the command line args for that batch
        """
        self.dimensions = dimensions
        self.interesting_systems_method = interesting_systems_method
        self.update_properties_method = update_properties_method
        self.configure_code_run = configure_code_run
        self.rejected_systems_method = rejected_systems_method
        self.batch_num = 0
        self.num_explored = 0
        self.finished = 0
        self.num_hits = 0
        self.fraction_explored = 1
        printProgressBar(0, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def explore(self, intial_pdf):
        """
        This function is the exploration phase of the stroopwafel
        IN:
            initial_pdf (NDimensionalDistribution) : An instance of NDimensionalDistribution showing how to sample from in the exploration phase
        """
        print("Exploration phase started")
        if not self.mc_only:
            self.prior_fraction_rejected = intial_pdf.calculate_rejection_rate(self.update_properties_method, self.rejected_systems_method, self.dimensions)
            print_logs(self.output_folder, "prior_fraction_rejected", self.prior_fraction_rejected)
        else:
            self.prior_fraction_rejected = 0
        while self.should_continue_exploring():
            batches = []
            for batch in range(self.num_batches_in_parallel):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                num_samples = int(2 * np.ceil(self.num_samples_per_batch / (1 - self.prior_fraction_rejected)))
                print("NUM SAMPLES---------------", num_samples)
                (locations, mask) = intial_pdf.run_sampler(num_samples)
                [location.revert_variables_to_original_scales() for location in locations]
                [location.properties.update({'generation': 0}) for location in locations]
                [location.properties.update({'gaussian': -1}) for location in locations]
                if self.update_properties_method != None:
                    self.update_properties_method(locations, self.dimensions)
                if self.rejected_systems_method != None:
                    rejected = self.rejected_systems_method(locations, self.dimensions)
                    print("rejected systems: ", rejected)   
                locations[:] = [location for location in locations if location.properties.get('is_rejected', 0) == 0]
                np.random.shuffle(locations)
                locations = locations[:self.num_samples_per_batch]
                [location.properties.pop('is_rejected', None) for location in locations]
                current_batch['samples'] = locations
                # generate_grid(locations, current_batch['grid_filename']) #for COMPAS
                grid = generate_grid_cosmic(locations) #for cosmic
                current_batch['grid'] = grid
                evol_start = time.time()
                command = self.configure_code_run(current_batch)
                evol_end = time.time()
                self.evolve_time += evol_end - evol_start
                # current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios) #for COMPAS
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            # self.process_batches(batches, True) #for COMPAS
            self.process_batches_cosmic(batches, True) #for cosmic
        self.num_hits_exploratory = self.num_hits
        print_logs(self.output_folder, "num_explored", self.num_explored)
        if self.mc_only:
            exit()
        else:
            print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(self.num_hits, self.num_explored, self.num_hits / self.num_explored, self.fraction_explored))

    def adapt(self, n_dimensional_distribution_type):
        """
        Adaptive phase of stroopwafel
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        if self.num_hits == 0:
            print ("No hits in the exploration phase\n")
            exit()
        hits = read_samples(self.output_filename, self.dimensions, only_hits = True)
        [location.transform_variables_to_new_scales() for location in hits]
        average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / len(self.dimensions))
        self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(hits, average_density_one_dim)
        for distribution in self.adapted_distributions:
            distribution.cov *= KAPPA * KAPPA
            distribution.alpha = 1 / len(self.adapted_distributions)
        print ("Adaptation phase finished!")

    def refine(self, n_dimensional_distribution_type):
        """
        Refinement phase of stroopwafel
        """
        self.entropies = []
        self.should_update = True
        for generation in range(NUM_GENERATIONS):
            samples = []
            self.distribution_rejection_rate = self.calculate_rejection_rate()
            self.num_samples_per_generation = int((self.total_num_systems - self.num_explored) / NUM_GENERATIONS)
            self.print_distributions(self.adapted_distributions, generation + 1)
            while self.num_samples_per_generation > 0 and self.finished < self.total_num_systems:
                batches = []
                for batch in range(min(self.num_batches_in_parallel, int(np.ceil(self.num_samples_per_generation / self.num_samples_per_batch)))):
                    current_batch = dict()
                    current_batch['number'] = self.batch_num
                    locations_ref = []
                    for index, distribution in enumerate(self.adapted_distributions):
                        num_samples = self.num_samples_per_batch * distribution.alpha
                        (locations, mask) = distribution.run_sampler(num_samples , self.dimensions, True)
                        [location.properties.update({'gaussian': index + 1}) for location in locations]
                        locations_ref.extend(np.asarray(locations)[mask])
                    [location.revert_variables_to_original_scales() for location in locations_ref]
                    if self.update_properties_method != None:
                        self.update_properties_method(locations_ref, self.dimensions)
                    if self.rejected_systems_method != None:
                        self.rejected_systems_method(locations_ref, self.dimensions)
                    locations_ref[:] = [location for location in locations_ref if location.properties.get('is_rejected', 0) == 0]
                    np.random.shuffle(locations_ref)
                    locations_ref = locations_ref[:self.num_samples_per_batch]
                    [location.properties.pop('is_rejected', None) for location in locations_ref]
                    current_batch['samples'] = locations_ref
                    [location.properties.update({'generation': generation + 1}) for location in locations_ref]
                    samples.extend(locations_ref)
                    grid = generate_grid_cosmic(locations_ref) #for cosmic
                    current_batch['grid'] = grid
                    evol_start = time.time()
                    command = self.configure_code_run(current_batch)
                    evol_end = time.time()
                    self.evolve_time += evol_end - evol_start
                    # current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                    batches.append(current_batch)
                    self.batch_num = self.batch_num + 1
                self.process_batches_cosmic(batches, False)
            if generation < NUM_GENERATIONS - 1 and self.should_update:
                self.update_distributions(samples, tolerance = 1e-10)
            if self.finished >= self.total_num_systems:
                break
        num_refined = self.total_num_systems - self.num_explored
        print_logs(self.output_folder, "total_num_systems", self.num_explored + num_refined)
        print ("\nRefinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(self.num_hits - self.num_hits_exploratory, num_refined, (self.num_hits - self.num_hits_exploratory) / num_refined))

    def process_batches(self, batches, is_exploration_phase):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
            is_exploration_phase (Boolean) : Whether the given batches come from exploration phase or not
        """
        for batch in batches:
            if batch['process']:
                returncode = batch['process'].wait()
            if not os.path.exists(os.path.join(self.output_folder, batch['output_container'])): #create batch folder
                os.makedirs(os.path.join(self.output_folder, batch['output_container']))
            folder = os.path.join(self.output_folder, batch['output_container'])
            shutil.move(batch['grid_filename'], os.path.join(folder, 'grid_' + str(batch['number']) + '.csv'))
            [location.properties.update({'is_hit': 0}) for location in batch['samples']]
            hits = 0
            if returncode >= 0 and self.interesting_systems_method is not None:
                hits = self.interesting_systems_method(batch)
            if (is_exploration_phase and not self.should_continue_exploring()) or self.finished >= self.total_num_systems or returncode < 0:
                #This batch is not needed anymore, delete the folder
                shutil.rmtree(os.path.join(self.output_folder, 'batch_' + str(batch['number'])))
                self.batch_num = self.batch_num - 1
                continue
            self.num_hits += hits
            self.finished += self.num_samples_per_batch
            print_samples(batch['samples'], self.output_filename, 'a')
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
            else:
                self.num_samples_per_generation -= self.num_samples_per_batch
            printProgressBar(self.finished, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)


    def process_batches_cosmic(self, batches, is_exploration_phase):
        for batch in batches:
            # if batch['process']:
            #     returncode = batch['process'].wait()
            # if not os.path.exists(os.path.join(self.output_folder, batch['output_container'])): #create batch folder
            #     os.makedirs(os.path.join(self.output_folder, batch['output_container']))
            # folder = os.path.join(self.output_folder, batch['output_container'])
            # shutil.move(batch['grid_filename'], os.path.join(folder, 'grid_' + str(batch['number']) + '.csv'))
            [location.properties.update({'is_hit': 0}) for location in batch['samples']]
           
            #Initialize columns in bpp and initC
            batch['bpp']['gaussian'] = -1
            batch['bpp']['generation'] = 0
            batch['bpp']['is_hit'] = 0
            batch['bpp']['mixture_weight'] = 0

            batch['initC']['gaussian'] = -1
            batch['initC']['generation'] = 0
            batch['initC']['is_hit'] = 0
            batch['initC']['mixture_weight'] = 0
            

            hits = 0
            hits = self.interesting_systems_method(batch)
            #update gaussian, generation, and is_hit
            for bin_num, location in enumerate(batch['samples']):
                batch['bpp'].loc[batch['bpp']['bin_num'] == bin_num, 'gaussian'] = location.properties['gaussian']
                batch['bpp'].loc[batch['bpp']['bin_num'] == bin_num, 'generation'] = location.properties['generation']
                batch['bpp'].loc[batch['bpp']['bin_num'] == bin_num, 'is_hit'] = location.properties['is_hit']
                batch['initC'].loc[batch['initC']['bin_num'] == bin_num, 'gaussian'] = location.properties['gaussian']
                batch['initC'].loc[batch['initC']['bin_num'] == bin_num, 'generation'] = location.properties['generation']
                batch['initC'].loc[batch['initC']['bin_num'] == bin_num, 'is_hit'] = location.properties['is_hit']

            #adjust bin num according to batch number
            batch['bpp']['bin_num'] += batch['number'] * self.num_samples_per_batch
            batch['initC']['bin_num'] += batch['number'] * self.num_samples_per_batch

            # Collect bpp and initC DataFrames from each batch
            self.all_bpp.append(batch['bpp'])
            self.all_initC.append(batch['initC'])
            self.num_hits += hits
            self.finished += self.num_samples_per_batch
            print_samples(batch['samples'], self.output_filename, 'a')
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
            else:
                self.num_samples_per_generation -= self.num_samples_per_batch
            printProgressBar(self.finished, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def update_distributions(self, locations, tolerance = 0):
        [location.transform_variables_to_new_scales() for location in locations]
        pi_norm = 1.0 / (1 - self.prior_fraction_rejected)
        q_norm = 1.0 / (1 - self.distribution_rejection_rate)
        samples = []
        mu = []
        sigma = []
        pi = []
        mask_hits = []
        alpha = []
        [samples.append(location.to_array()) for location in locations]
        [pi.append(location.calculate_prior_probability() * pi_norm) for location in locations]
        [mu.append(distribution.mean.to_array()) for distribution in self.adapted_distributions]
        [sigma.append(distribution.cov) for distribution in self.adapted_distributions]
        [alpha.append(distribution.alpha) for distribution in self.adapted_distributions]
        [mask_hits.append(location.properties['is_hit']) for location in locations]
        samples = np.asarray(samples)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        pi = np.asarray(pi)
        mask_hits = np.asarray(mask_hits)
        num_distributions = len(mu)
        num_samples = len(samples)
        xPDF = np.zeros((num_distributions, num_samples))
        for i in range(num_distributions):
            xPDF[i, :] = multivariate_normal.pdf(samples, mu[i], sigma[i], allow_singular = True)
        xPDF = xPDF.T
        qPDF = xPDF * alpha * q_norm
        rho = qPDF / np.sum(qPDF, axis = 1)[:, None]
        gaussian_weights = np.asarray((pi * mask_hits) / np.sum(qPDF, axis = 1))
        weights_normalized = (gaussian_weights / np.sum(gaussian_weights))[:, None]
        alpha = np.sum(weights_normalized * rho, axis = 0)
        insignificant_components = np.argwhere(alpha <= tolerance)
        alpha = np.delete(alpha, insignificant_components)
        for index in range(len(self.dimensions)):
            mu[:, index] = np.sum(weights_normalized * samples[:, index][:, None] * rho, axis = 0)
        mu = np.delete(mu, insignificant_components, axis = 0)
        mu = mu / alpha[:, None]
        sigma = np.delete(sigma, insignificant_components, axis = 0)
        for i in range(len(mu)):
            distance = np.asarray(mu[i] - samples)[:, :, None]
            matrix = np.einsum('nij,nji->nij', distance, distance)
            factor = weights_normalized[:, 0] * rho[:, i]
            sigma[i] = np.sum(factor[:, None, None] * matrix, axis = 0) / alpha[i]
        #check entropy change, maybe we already reached maximum
        entropy_change = np.exp(entropy(weights_normalized)) / num_samples
        if len(self.entropies) >= 1 and entropy_change - self.entropies[-1] < MIN_ENTROPY_CHANGE:
            #this is not a good update, probably the last one was the best, so lets revert to it
            print (entropy_change)
            generation_to_revert = len(self.entropies)
            self.adapted_distributions = self.read_distributions(generation_to_revert)
            self.should_update = False
            return
        self.adapted_distributions = self.adapted_distributions[:len(alpha)]
        for index, distribution in enumerate(self.adapted_distributions):
            for i, dimension in enumerate(sorted(distribution.mean.dimensions.keys(), key = lambda d: d.name)):
                distribution.mean.dimensions[dimension] = mu[index][i]
            distribution.cov = sigma[index]
            distribution.alpha = alpha[index]
        print_logs(self.output_folder, "p", entropy_change)
        self.entropies.append(entropy_change)
        # self.add_original_forgotten_distributions()

    def calculate_weights_of_samples(self):
        reading_start = time.time()
        locations = read_samples(self.output_filename, self.dimensions)
        reading_end = time.time()
        [location.transform_variables_to_new_scales() for location in locations]
        pi_norm = 1.0 / (1 - self.prior_fraction_rejected)
        pi = []
        [pi.append(location.calculate_prior_probability() * pi_norm) for location in locations]
        pi = np.asarray(pi)
        num_samples = len(locations)
        samples = []
        [samples.append(location.to_array()) for location in locations]
        samples = np.asarray(samples)
        fraction_explored = self.num_explored / float(num_samples)
        den = np.ones(num_samples) * (fraction_explored) * pi
        gen_loop_start = time.time()
        for generation in range(NUM_GENERATIONS):
            distributions = self.read_distributions(generation + 1)
            num_distributions = len(distributions)
            if num_distributions == 0:
                continue
            mu = []
            sigma = []
            alpha = []
            [mu.append(distribution.mean.to_array()) for distribution in distributions]
            [sigma.append(distribution.cov) for distribution in distributions]
            [alpha.append(distribution.alpha) for distribution in distributions]
            xPDF = np.zeros((num_distributions, num_samples))
            for i in range(num_distributions):
                xPDF[i, :] = multivariate_normal.pdf(samples, mu[i], sigma[i], allow_singular = True)
            xPDF = xPDF.T
            q_norm = 1 / (1 - distributions[0].rejection_rate)
            q_PDF = xPDF * np.asarray(alpha)
            den += (np.sum(q_PDF, axis = 1) * (1 - fraction_explored) * q_norm) / NUM_GENERATIONS
        gen_loop_end = time.time()
        weights = pi / den
        [location.properties.update({'mixture_weight' : weights[index]}) for index, location in enumerate(locations)]
        [location.revert_variables_to_original_scales() for location in locations]
        print_samples(locations, self.output_filename, 'w')
        # Concatenate bpp and initC DataFrames from all batches
        concat_start = time.time()
        full_bpp = pd.concat(self.all_bpp, ignore_index=True)
        full_initC = pd.concat(self.all_initC, ignore_index=True)
        concat_end = time.time()

        # Update gaussian, generation, is_hit, and mixture_weight in full_bpp and full_initC
        update_start = time.time()
        # for bin_num in range(len(locations)):
        #     full_bpp.loc[full_bpp['bin_num'] == bin_num, 'mixture_weight'] = weights[bin_num]
        #     full_initC.loc[full_initC['bin_num'] == bin_num, 'mixture_weight'] = weights[bin_num]
        bin_nums = range(len(locations))
        full_bpp.loc[full_bpp['bin_num'].isin(bin_nums), 'mixture_weight'] = full_bpp['bin_num'].map(dict(zip(bin_nums, weights)))
        full_initC.loc[full_initC['bin_num'].isin(bin_nums), 'mixture_weight'] = full_initC['bin_num'].map(dict(zip(bin_nums, weights)))
        update_end = time.time()

        print(full_bpp)
        print(full_initC)

        # Save full_bpp and full_initC to h5 files
        save_start = time.time()
        full_bpp.to_hdf(os.path.join(self.output_folder, self.cosmic_filename), key='bpp', mode='w')
        full_initC.to_hdf(os.path.join(self.output_folder, self.cosmic_filename), key='initC', mode='a')
        save_end = time.time()
        print("Evolve time: ", self.evolve_time)
        print("Reject time: ", self.reject_time)
        print("Concat time: ", concat_end - concat_start)
        print("Save time: ", save_end - save_start)
        print("Gen loop time: ", gen_loop_end - gen_loop_start)
        print("Update time: ", update_end - update_start)
        print("Reading time: ", reading_end - reading_start)

    def calculate_rejection_rate(self):
        fractional_rejected = 0
        N_GAUSS = 10000
        # N_GAUSS = 1000
        reject_start = time.time()
        print("Calculating rejection rate")
        print("Number of gaussians: ", len(self.adapted_distributions))
        # for distribution in self.adapted_distributions:
        #     (locations, mask) = distribution.run_sampler(N_GAUSS, self.dimensions)
        #     rejected = N_GAUSS - np.sum(mask)
        #     locations = np.asarray(locations)[mask]
        #     print("Locations: ", len(locations))
        #     [location.revert_variables_to_original_scales() for location in locations]
        #     self.update_properties_method(locations, self.dimensions)
        #     rejected += self.rejected_systems_method(locations, self.dimensions)
        #     fractional_rejected += rejected * distribution.alpha / N_GAUSS
        with MultiPool() as pool:
            fractional_rejected = np.sum(pool.map(self.dist_rejection_rate, self.adapted_distributions))
        reject_end = time.time()
        self.reject_time += reject_end - reject_start
        return fractional_rejected
    
    def dist_rejection_rate(self, distribution):
        N_GAUSS = 10000
        (locations, mask) = distribution.run_sampler(N_GAUSS, self.dimensions)
        rejected = N_GAUSS - np.sum(mask)
        locations = np.asarray(locations)[mask]
        [location.revert_variables_to_original_scales() for location in locations]
        self.update_properties_method(locations, self.dimensions)
        rejected += self.rejected_systems_method(locations, self.dimensions)
        return rejected * distribution.alpha / N_GAUSS

    def print_distributions(self, distributions, generation_number):
        num_distributions = len(distributions)
        with open(os.path.join(self.output_folder, "distributions_" + str(generation_number) + ".txt"), 'w') as file:
            file.write("%d\n"%(generation_number))
            file.write("%d\n"%(num_distributions))
            file.write("%d\n"%(len(self.dimensions)))
            file.write("%f\n"%(self.distribution_rejection_rate))
            for distribution in distributions:
                file.write("%f\n"%(distribution.alpha))
                file.write("\t".join(str(i) for i in distribution.mean.to_array()))
                file.write("\n")
                for index in range(len(self.dimensions)):
                    file.write("\t".join(str(i) for i in distribution.cov[index]))
                    file.write("\n")

    def read_distributions(self, generation_number):
        try:
            with open(os.path.join(self.output_folder, "distributions_" + str(generation_number) + ".txt"), 'r') as file:
                generation_number = int(file.readline())
                num_distributions = int(file.readline())
                num_dimensions = int(file.readline())
                distribution_rejection_rate = float(file.readline())
                distributions = []
                for index in range(num_distributions):
                    alpha = float(file.readline())
                    mean_values = [float(val) for val in file.readline().split("\t")]
                    cov = [[] for i in range(num_dimensions)]
                    for i in range(num_dimensions):
                        cov[i] = [float(val) for val in file.readline().split("\t")]
                    means = dict()
                    for index, dimension in enumerate(sorted(self.dimensions, key = lambda d: d.name)):
                        means[dimension] = mean_values[index]
                    gaussian = Gaussian(Location(means, {}), cov = cov, alpha = alpha)
                    gaussian.rejection_rate = distribution_rejection_rate
                    distributions.append(gaussian)
                return distributions
        except Exception as error:
            return []
