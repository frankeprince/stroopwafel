#!/usr/bin/env python

import os
import pandas as pd
import shutil
import time
import numpy as np
from modules import *

start_time = time.time()

#Define the parameters to the constructor of stroopwafel
NUM_DIMENSIONS = 9 #Number of dimensions you want to samples
NUM_BINARIES = 1000000 #total number of systems
NUM_BATCHES = 50 #Number of batches you want to run in parellel
NUM_SAMPLES_PER_BATCH = 1000 #Number of samples generated by each of the batch
debug = False #If True, will generate the logs given by the external program (like COMPAS)
run_on_helios = False #If True, it will run on a clustered system helios, rather than your pc
mc_only = False

compas_executable = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/COMPAS') # Location of the executable
output_folder = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/output') # Where you want to receieve outputs
if os.path.exists(output_folder):
    command = input ("The output folder already exists. If you continue, I will remove all its content. Press (Y/N)\n")
    if (command == 'Y'):
        shutil.rmtree(output_folder)
    else:
        exit()
os.makedirs(output_folder)

# STEP 1 : Create an instance of the Stroopwafel class
sw = stroopwafel.Stroopwafel(NUM_DIMENSIONS, NUM_BINARIES, NUM_BATCHES, NUM_SAMPLES_PER_BATCH, output_folder, debug = debug, run_on_helios = run_on_helios, mc_only = mc_only)

# STEP 2 : Define the functions
def create_dimensions():
    """
    This Function that will create all the dimensions for stroopwafel, a dimension is basically one of the variables you want to sample
    Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in stroopwafel.py for more.
    It takes the name of the dimension, its max and min value. 
    The Sampler class (also in stroopwafel) will tell how to sample this dimension. Similarly, prior tells it how it calculates the prior
    OUT:
        As Output, this should return a list containing all the instances of Dimension class.
    """
    m1 = classes.Dimension('Mass_1', 5, 50, sampler.kroupa, prior.kroupa)
    q = classes.Dimension('q', 0.25, 1, sampler.uniform, prior.uniform, should_print = False)
    a = classes.Dimension('Separation', 0.1, 40, sampler.flat_in_log, prior.flat_in_log)
    kick_velocity_random_1 = classes.Dimension('Kick_Velocity_Random_1', 0, 1, sampler.uniform, prior.uniform)
    kick_theta_1 = classes.Dimension('Kick_Theta_1', -np.pi / 2, np.pi / 2, sampler.uniform_in_sine, prior.uniform_in_sine)
    kick_phi_1 = classes.Dimension('Kick_Phi_1', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    kick_velocity_random_2 = classes.Dimension('Kick_Velocity_Random_2', 0, 1, sampler.uniform, prior.uniform)
    kick_theta_2 = classes.Dimension('Kick_Theta_2', -np.pi / 2, np.pi / 2, sampler.uniform_in_sine, prior.uniform_in_sine)
    kick_phi_2 = classes.Dimension('Kick_Phi_2', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    return [m1, q, a, kick_velocity_random_1, kick_theta_1, kick_phi_1, kick_velocity_random_2, kick_theta_2, kick_phi_2]

def update_properties(locations):
    """
    This function is not mandatory, it is required only if you have some dependent variable. 
    For example, if you want to sample Mass_1 and q, then Mass_2 is a dependent variable which is product of the two.
    Similarly, you can assume that Metallicity_2 will always be equal to Metallicity_1
    IN:
        locations (list(Location)) : A list containing objects of Location class in stroopwafel.py. 
        You can play with them and update whatever fields you like or add more in the property (which is a dictionary) of Location
    OUT: Not Required
    """
    m1 = dimensions[0]
    q = dimensions[1]
    for location in locations:
        location.properties['Mass_2'] = location.dimensions[m1] * location.dimensions[q]
        location.properties['Metallicity_2'] = location.properties['Metallicity_1'] = 0.0142
        location.properties['Eccentricity'] = 0
        location.properties['Kick_Mean_Anomaly_1'] = np.random.uniform(0, 2 * np.pi, 1)[0]
        location.properties['Kick_Mean_Anomaly_2'] = np.random.uniform(0, 2 * np.pi, 1)[0]

def configure_code_run(batch):
    """
    This function tells stroopwafel what program to run, along with its arguments.
    IN:
        batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
            It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
            for each batch run in this dictionary. For example, here I have stored the 'system_params_filename' so that I can read them during discovery of interesting systems below
    OUT:
        compas_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
        Additionally one must also store the grid_filename in the batch so that the grid file is created
    """
    batch_num = batch['number']
    grid_filename = output_folder + '/grid_' + str(batch_num) + '.txt'
    output_container = 'batch_' + str(batch_num)
    compas_args = [compas_executable, "--grid", grid_filename, '--outputPath', output_folder, '--logfile-delimiter', 'COMMA', '--output-container', output_container, '--random-seed', np.random.randint(2, 2**63 - 1)]
    batch['grid_filename'] = grid_filename
    batch['output_container'] = output_container
    return compas_args

def interesting_systems(batch):
    """
    This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
    IN:
        batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
    OUT:
        list(Location): A list of Location objects which a user defines as interesting.
        In the below example, I define all the DCOs as interesting, so I read the files, get the parameters from the system_params file and create 
        Location object for each of them with the dimensions and the properties.
    """
    try:
        folder = os.path.join(output_folder, batch['output_container'])
        double_compact_objects = pd.read_csv(folder + '/BSE_Double_Compact_Objects.csv', skiprows = 2)
        double_compact_objects.rename(columns = lambda x: x.strip(), inplace=True)
        double_compact_objects.set_index('ID')
        double_compact_objects.rename(columns = {'Mass_1': 'Mass@DCO_1', 'Mass_2': 'Mass@DCO_2'}, inplace = True)
        dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 13, double_compact_objects['Stellar_Type_2'] == 13)]
        system_parameters = pd.read_csv(folder + '/BSE_System_Parameters.csv', skiprows = 2)
        system_parameters.rename(columns = lambda x: x.strip(), inplace=True)
        system_parameters.set_index('ID')
        system_parameters.rename(columns = {'Mass@ZAMS_1': 'Mass_1', 'Mass@ZAMS_2': 'Mass_2', 'Separation@ZAMS' : 'Separation',\
            'Eccentricity@ZAMS': 'Eccentricity', 'SN_Kick_Magnitude_Random_Number_1': 'Kick_Velocity_Random_1', 'SN_Kick_Theta_1': 'Kick_Theta_1',\
            'SN_Kick_Phi_1': 'Kick_Phi_1', 'SN_Kick_Mean_Anomaly_1': 'Kick_Mean_Anomaly_1', 'SN_Kick_Magnitude_Random_Number_2': 'Kick_Velocity_Random_2',\
            'SN_Kick_Theta_2': 'Kick_Theta_2', 'SN_Kick_Phi_2': 'Kick_Phi_2', 'SN_Kick_Mean_Anomaly_2': 'Kick_Mean_Anomaly_2',\
            'Metallicity@ZAMS_1': 'Metallicity_1', 'Metallicity@ZAMS_2': 'Metallicity_2'}, inplace = True)
        systems = system_parameters[np.isin(system_parameters['ID'], dns['ID'])]
        systems = pd.merge(systems, dns, on=['ID'])
        locations = []
        for index, system in systems.iterrows():
            location = dict()
            for dimension in dimensions:
                if dimension.name == 'q':
                    location[dimension] = system['Mass_2'] / system['Mass_1']
                else:
                    location[dimension] = system[dimension.name]
            properties = dict()
            properties['batch'] = batch['number']
            for prop in ('ID', 'Metallicity_2', 'Mass_2', 'Eccentricity'):
                properties[prop] = system[prop]
            locations.append(classes.Location(location, properties))
        return locations
    except IOError as error:
        return []

def selection_effects(sw):
    """
    Fills in selection effects for each of the distributions
    IN:
        sw (Stroopwafel) : Stroopwafel object
    """
    #find means of masses
    if hasattr(sw, 'adapted_distributions'):
        biased_masses = []
        for distribution in sw.adapted_distributions:
            biased_masses.append(np.power(max([distribution.mean.properties['Mass@DCO_1'], distribution.mean.properties['Mass@DCO_2']]), 2.2))
        # update the weights
        mean = np.mean(biased_masses)
        for distribution in sw.adapted_distributions:
            distribution.biased_weight = np.power(max([distribution.mean.properties['Mass@DCO_1'], distribution.mean.properties['Mass@DCO_2']]), 2.2) / mean

#STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
sw.initialize(interesting_systems, configure_code_run, update_properties_method = update_properties)
dimensions = create_dimensions()
intial_pdf = distributions.InitialDistribution(dimensions)
#STEP 4: Run the 4 phases of stroopwafel
sw.explore(dimensions, intial_pdf) #Pass in the dimensions list created, and the initial distribution for exploration phase
sw.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
## Do selection effects
#selection_effects(sw)
sw.refine() #Stroopwafel will draw samples from the adapted distributions
sw.postprocess(output_folder + "/hits.csv") #Run it to create weights of the hits found. Pass in a filename to store all the hits

end_time = time.time()
print ("Total running time = %d seconds" %(end_time - start_time))
