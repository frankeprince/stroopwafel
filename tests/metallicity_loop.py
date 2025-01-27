#!/usr/bin/env python

import os
import pandas as pd
import shutil
import time
import numpy as np
import sys
sys.path.append('../') #Only required in the test directory for testing purposes
from stroopwafel import genais, cosmic, classes, prior, sampler, distributions, constants, utils
import argparse
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
from astropy import constants as const
import astropy.units as u
print(sys.path)
parser=argparse.ArgumentParser()
parser.add_argument('--num_systems', help = 'Total number of systems', type = int, default = 10000)
parser.add_argument('--num_cores', help = 'Number of cores to run in parallel', type = int, default = 1)
parser.add_argument('--num_per_core', help = 'Number of systems to generate in one core', type = int, default = 100)
parser.add_argument('--debug', help = 'If debug of COMPAS is to be printed', type = bool, default = False)
parser.add_argument('--mc_only', help = 'If run in MC simulation mode only', type = bool, default = False)
parser.add_argument('--run_on_helios', help = 'If we are running on helios (or other slurm) nodes', type = bool, default = False)
parser.add_argument('--output_filename', help = 'Output filename', default = 'samples.csv')
parser.add_argument('--pairs', help = "Pairs to select for", default = 'all')
namespace, extra_params = parser.parse_known_args()

# STEP 2 : Define the functions
def create_dimensions():
    """
    This Function that will create all the dimensions for stroopwafel, a dimension is basically one of the variables you want to sample
    Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in classes.py for more.
    It takes the name of the dimension, its max and min value. 
    The Sampler class will tell how to sample this dimension. Similarly, prior tells it how it calculates the prior. You can find more of these in their respective modules
    OUT:
        As Output, this should return a list containing all the instances of Dimension class.
    """
    m1 = classes.Dimension('Mass_1', 0.7, 150, sampler.kroupa, prior.kroupa)
    q = classes.Dimension('q', 0, 1, sampler.uniform, prior.uniform)
    porb = classes.Dimension('Porb', 0.15, 5.5, sampler.sana, prior.sana)
    ecc = classes.Dimension('Eccentricity', 0.0001, 0.9, sampler.sana_ecc, prior.sana_ecc)
    return [m1, q, porb, ecc]
    # porb = classes.Dimension('Porb', 100, 10000, sampler.uniform, prior.uniform)
    # a = classes.Dimension('Separation', 1, 100, sampler.flat_in_log, prior.flat_in_log) #try different values here?
    # a = classes.Dimension('Separation', 1, 100, sampler.uniform, prior.uniform) #temp change for simplicity
    #kick_velocity_random_1 = classes.Dimension('Kick_Velocity_Random_1', 0, 1, sampler.uniform, prior.uniform)
    #kick_theta_1 = classes.Dimension('Kick_Theta_1', -np.pi / 2, np.pi / 2, sampler.uniform_in_cosine, prior.uniform_in_cosine)
    #kick_phi_1 = classes.Dimension('Kick_Phi_1', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    #kick_velocity_random_2 = classes.Dimension('Kick_Velocity_Random_2', 0, 1, sampler.uniform, prior.uniform)
    #kick_theta_2 = classes.Dimension('Kick_Theta_2', -np.pi / 2, np.pi / 2, sampler.uniform_in_cosine, prior.uniform_in_cosine)
    #kick_phi_2 = classes.Dimension('Kick_Phi_2', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    #return [m1, q, a, kick_velocity_random_1, kick_theta_1, kick_phi_1, kick_velocity_random_2, kick_theta_2, kick_phi_2]
    

def update_properties(locations, dimensions):
    """
    This function is not mandatory, it is required only if you have some dependent variable. 
    For example, if you want to sample Mass_1 and q, then Mass_2 is a dependent variable which is product of the two.
    Similarly, you can assume that Metallicity_2 will always be equal to Metallicity_1
    IN:
        locations (list(Location)) : A list containing objects of Location class in classes.py. 
        You can play with them and update whatever fields you like or add more in the property (which is a dictionary)
    OUT: Not Required
    """
    m1 = dimensions[0]
    q = dimensions[1]
    porb = dimensions[2]
    ecc = dimensions[3]
    for location in locations:
        location.properties['Mass_2'] = location.dimensions[m1] * location.dimensions[q]
        location.properties['Metallicity_2'] = location.properties['Metallicity_1'] = metallicity
        location.properties['Separation'] = ((location.dimensions[porb] ** 2) * (location.dimensions[m1] + location.properties['Mass_2'])) ** (1 / 3)
        # location.properties["Eccentricity"] = 0
        #location.properties['Kick_Mean_Anomaly_1'] = np.random.uniform(0, 2 * np.pi, 1)[0]
        #location.properties['Kick_Mean_Anomaly_2'] = np.random.uniform(0, 2 * np.pi, 1)[0]

def configure_code_run(batch):
    """
    This function tells stroopwafel what program to run, along with its arguments.
    IN:
        batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
            It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
            for each batch run in this dictionary. For example, here I have stored the 'output_container' and 'grid_filename' so that I can read them during discovery of interesting systems below
    OUT:
        compas_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
        Additionally one must also store the grid_filename in the batch so that the grid file is created
    """
    batch_num = batch['number']
    grid_filename = os.path.join(output_folder,'grid_' + str(batch_num) + '.csv')
    output_container = 'batch_' + str(batch_num)
    # compas_args = [compas_executable, "--grid", '"' + grid_filename + '"', '--outputPath', '"' + output_folder + '"', '--logfile-delimiter', 'COMMA', '--output-container', output_container, '--random-seed', np.random.randint(2, 2**63 - 1)]
    compas_args = []
    batch['grid_filename'] = grid_filename
    batch['output_container'] = output_container
    return compas_args

def configure_code_run_cosmic(batch):
    """
    This function tells stroopwafel how to evolve binaries with cosmic
    """
    pop_length = len(batch['samples'])
    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1}
    batch_initial = InitialBinaryTable.InitialBinaries(m1=batch['grid']['Mass_1'], m2=batch['grid']['Mass_2'], porb=batch['grid']['Porb'], 
                                                       ecc=batch['grid']['Eccentricity'], tphysf=np.full(pop_length, 13700), 
                                                       kstar1=np.full(pop_length, 1), kstar2=np.full(pop_length, 1), 
                                                       metallicity=batch['grid']['Metallicity_1'])
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=batch_initial, BSEDict=BSEDict, nproc=NUM_CPU_CORES)
    batch['bpp'] = bpp
    batch['bcm'] = bcm
    batch['initC'] = initC
    batch['kick_info'] = kick_info

def interesting_systems(batch):
    """
    This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
    IN:
        batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
    OUT:
        Number of interesting systems
        In the below example, I define all the NSs as interesting, so I read the files, get the SEED from the system_params file and define the key is_hit in the end for all interesting systems 
    """
    try:
        folder = os.path.join(output_folder, batch['output_container'])
        system_parameters = pd.read_csv(os.path.join(folder, 'BSE_System_Parameters.csv'), skiprows = 2)
        system_parameters.rename(columns = lambda x: x.strip(), inplace = True)
        seeds = system_parameters['SEED']
        for index, sample in enumerate(batch['samples']):
            seed = seeds[index]
            sample.properties['SEED'] = seed
            sample.properties['is_hit'] = 0
            sample.properties['batch'] = batch['number']
        double_compact_objects = pd.read_csv(os.path.join(folder, 'BSE_Double_Compact_Objects.csv'), skiprows = 2)
        double_compact_objects.rename(columns = lambda x: x.strip(), inplace = True)
        #Generally, this is the line you would want to change.
        dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 14, double_compact_objects['Stellar_Type_2'] == 14)]
        interesting_systems_seeds = set(dns['SEED'])
        for sample in batch['samples']:
            if sample.properties['SEED'] in interesting_systems_seeds:
                sample.properties['is_hit'] = 1
        return len(dns)
    except IOError as error:
        return 0

def interesting_systems_cosmic(batch):
    """
    This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
    IN:
        batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
    OUT:
        Number of interesting systems
        In the below example, I define all the NSs as interesting, so I read the files, get the SEED from the system_params file and define the key is_hit in the end for all interesting systems 
    """
    k_select_dict = {'NSNS': ([13],[13]), 'BHNS': ([14], [13]), 'BHBH': ([14],[14]), 'BHWD': ([14],[10,11,12]), 'NSWD': ([13], [10, 11, 12]), 'all': ([10, 11, 12, 13, 14], [10, 11, 12, 13, 14]) }
    k_select = k_select_dict[pair]
    k1_select = k_select[0]
    k2_select = k_select[1]
    bpp = batch["bpp"]
    pairs_mask = ((bpp.kstar_1.isin(k1_select)) & (bpp.kstar_2.isin(k2_select))) | ((bpp.kstar_1.isin(k2_select)) & (bpp.kstar_2.isin(k1_select)))
    interesting_systems_mask = pairs_mask & (bpp.sep > 0)
    interesting_systems_table = bpp.loc[interesting_systems_mask].drop_duplicates(subset = 'bin_num', keep = 'first') #faster
    bin_nums = interesting_systems_table.index
    for sample in batch['samples']:
        sample.properties['is_hit'] = 0
    for bin_num in bin_nums:
        batch["samples"][bin_num].properties['is_hit'] = 1
    print("HITS: ", len(bin_nums))
    return len(interesting_systems_table)

    # interesting_systems_mask = bpp.kstar_1.isin([10,11,12,13,14]) & bpp.kstar_2.isin([10,11,12,13,14])

    # interesting_systems_table = bpp.loc[interesting_systems_mask].groupby('bin_num').first() #slower

    #     folder = os.path.join(output_folder, batch['output_container'])
    #     system_parameters = pd.read_csv(os.path.join(folder, 'BSE_System_Parameters.csv'), skiprows = 2)
    #     system_parameters.rename(columns = lambda x: x.strip(), inplace = True)
    #     seeds = system_parameters['SEED']
    #     for index, sample in enumerate(batch['samples']):
    #         seed = seeds[index]
    #         sample.properties['SEED'] = seed
    #         sample.properties['is_hit'] = 0
    #         sample.properties['batch'] = batch['number']
    #     double_compact_objects = pd.read_csv(os.path.join(folder, 'BSE_Double_Compact_Objects.csv'), skiprows = 2)
    #     double_compact_objects.rename(columns = lambda x: x.strip(), inplace = True)
    #     #Generally, this is the line you would want to change.
    #     dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 14, double_compact_objects['Stellar_Type_2'] == 14)]
    #     interesting_systems_seeds = set(dns['SEED'])
    #     for sample in batch['samples']:
    #         if sample.properties['SEED'] in interesting_systems_seeds:
    #             sample.properties['is_hit'] = 1
    #     return len(dns)
    # except IOError as error:
    #     return 0

def rejected_systems(locations, dimensions):
    """
    This method takes a list of locations and marks the systems which can be
    rejected by the prior distribution
    IN:
        locations (List(Location)): list of location to inspect and mark rejected
    OUT:
        num_rejected (int): number of systems which can be rejected
    """
    m1 = dimensions[0]
    q = dimensions[1]
    # a = dimensions[2]
    porb = dimensions[2]
    ecc = dimensions[3]
    mass_1 = [location.dimensions[m1] for location in locations]
    mass_2 = [location.properties['Mass_2'] for location in locations]
    metallicity_1 = [location.properties['Metallicity_1'] for location in locations]
    metallicity_2 = [location.properties['Metallicity_2'] for location in locations]
    eccentricity = [location.dimensions[ecc] for location in locations]
    # eccentricity = [location.properties['Eccentricity'] for location in locations]
    num_rejected = 0
    for index, location in enumerate(locations):
        radius_1 = utils.get_zams_radius(mass_1[index], metallicity_1[index])
        radius_2 = utils.get_zams_radius(mass_2[index], metallicity_2[index])
        roche_lobe_tracker_1 = radius_1 / (location.properties['Separation'] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_1[index], mass_2[index]))
        roche_lobe_tracker_2 = radius_2 / (location.properties['Separation'] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_2[index], mass_1[index]))
        location.properties['is_rejected'] = 0
        if (mass_2[index] < constants.MINIMUM_SECONDARY_MASS) or (location.properties['Separation'] <= (radius_1 + radius_2)) \
        or roche_lobe_tracker_1 > 1 or roche_lobe_tracker_2 > 1:
            location.properties['is_rejected'] = 1
            num_rejected += 1
    return num_rejected

# def rejected_systems_cosmic(locations, dimensions):
#     """
#     This method takes a list of locations and marks the systems which can be
#     rejected by the prior distribution
#     IN:
#         locations (List(Location)): list of location to inspect and mark rejected
#     OUT:
#         num_rejected (int): number of systems which can be rejected
#     """
#     m1 = dimensions[0]
#     q = dimensions[1]
#     a = dimensions[2]
#     mass_1 = [location.dimensions[m1] for location in locations]
#     mass_2 = [location.properties['Mass_2'] for location in locations]
#     porb = [location.properties['Porb'] for location in locations]
#     metallicity_1 = [location.properties['Metallicity_1'] for location in locations]
#     metallicity_2 = [location.properties['Metallicity_2'] for location in locations]
#     eccentricity = [location.properties['Eccentricity'] for location in locations]
#     num_rejected = 0
#     batch_initial = InitialBinaryTable.InitialBinaries(m1=mass_1, m2=mass_2, porb=porb, ecc=eccentricity, 
#                                                        tphysf=np.full(len(locations), 0), kstar1=np.full(len(locations), 1), kstar2=np.full(len(locations), 1), 
#                                                        metallicity=metallicity_1)
#     print("EVOLVING")
#     bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=batch_initial, BSEDict=BSEDict, nproc=NUM_CPU_CORES)
#     print("EVOLVED")
#     rejected_systems = bpp.groupby('bin_num', as_index=False).first()
#     rejected_systems = rejected_systems.loc[rejected_systems['RRLO_1'] > 1]
#     print("REJECTING")
#     for bin_num in rejected_systems['bin_num']:
#         locations[bin_num].properties['is_rejected'] = 1
#         num_rejected += 1
#     # for index, location in enumerate(locations):
#     #     radius_1 = utils.get_zams_radius(mass_1[index], metallicity_1[index])
#     #     radius_2 = utils.get_zams_radius(mass_2[index], metallicity_2[index])
#     #     roche_lobe_tracker_1 = radius_1 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_1[index], mass_2[index]))
#     #     roche_lobe_tracker_2 = radius_2 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_2[index], mass_1[index]))
#     #     location.properties['is_rejected'] = 0
#     #     if (mass_2[index] < constants.MINIMUM_SECONDARY_MASS) or (location.dimensions[a] <= (radius_1 + radius_2)) \
#     #     or roche_lobe_tracker_1 > 1 or roche_lobe_tracker_2 > 1:
#     #         location.properties['is_rejected'] = 1
#     #         num_rejected += 1
#     return num_rejected

if __name__ == '__main__':
    
    # metallicities = np.logspace(-4, np.log10(0.03), 50)
    metallicities = [0.0142]
    # pairs = ['BHWD', 'NSNS', 'BHNS', 'BHBH', 'NSWD']
    pairs = ['all']
    start_time = time.time()
    #Define the parameters to the constructor of stroopwafel
    TOTAL_NUM_SYSTEMS = namespace.num_systems #total number of systems you want in the end
    NUM_CPU_CORES = namespace.num_cores #Number of cpu cores you want to run in parellel
    NUM_SYSTEMS_PER_RUN = namespace.num_per_core #Number of systems generated by each of run on each cpu core
    debug = namespace.debug #If True, will print the logs given by the external program (like COMPAS)
    run_on_helios = namespace.run_on_helios #If True, it will run on a clustered system helios, rather than your pc
    mc_only = namespace.mc_only # If you dont want to do the refinement phase and just do random mc exploration
    output_filename = namespace.output_filename #The name of the output file
    pair = namespace.pairs #The type of pairs to select for
    # compas_executable = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/COMPAS') # Location of the executable
    
    for pair in pairs:
        for metallicity in metallicities:
            pair = pair
            metallicity = metallicity

            cosmic_filename = pair + '_' + str(metallicity)[2:] + '.h5'
            print(cosmic_filename)
            output_folder =  os.path.join(os.getcwd(), 'output/' + pair + '/' + str(metallicity)) # Folder you want to receieve outputs, here the current working directory, but you can specify anywhere
            if os.path.exists(output_folder):
                if NUM_CPU_CORES > 1:
                    shutil.rmtree(output_folder)
                else:
                    command = input ("The output folder already exists. If you continue, I will remove all its content. Press (Y/N)\n")
                    if (command == 'Y'):
                        shutil.rmtree(output_folder)
                    else:
                        exit()
            os.makedirs(output_folder)

            # STEP 1 : Create an instance of the Stroopwafel class
            cosmic_object = cosmic.Cosmic(TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_RUN, output_folder, output_filename, cosmic_filename, debug = debug, run_on_helios = run_on_helios, mc_only = mc_only)


            #STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
            dimensions = create_dimensions()
            cosmic_object.initialize(dimensions, interesting_systems_cosmic, configure_code_run_cosmic, rejected_systems, update_properties_method = update_properties)

            intial_pdf = distributions.InitialDistribution(dimensions)
            #STEP 4: Run the 4 phases of stroopwafel
            explore_start = time.time()
            cosmic_object.explore(intial_pdf) #Pass in the initial distribution for exploration phase
            explore_end = time.time()
            adapt_start = time.time()
            cosmic_object.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
            adapt_end = time.time()
            refine_start = time.time()
            cosmic_object.refine(n_dimensional_distribution_type = distributions.Gaussian) #Stroopwafel will draw samples from the adapted distributions
            refine_end = time.time()
            weights_start = time.time()
            cosmic_object.calculate_weights_of_samples()
            weights_end = time.time()

            end_time = time.time()
            print("--------------COMPLETE--------")
            print("Number of cores = %d" %NUM_CPU_CORES)
            print("Number of systems = %d" %TOTAL_NUM_SYSTEMS)
            print("Systems per batch = %d" %NUM_SYSTEMS_PER_RUN)
            print ("Exploration time = %d seconds" %(explore_end - explore_start))
            print ("Adaptation time = %d seconds" %(adapt_end - adapt_start))
            print ("Refinement time = %d seconds" %(refine_end - refine_start))
            print ("Weight Calculation time = %d seconds" %(weights_end - weights_start))
            print ("Total running time = %d seconds" %(end_time - start_time))
