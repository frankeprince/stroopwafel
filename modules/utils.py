import numpy as np
import os
import subprocess
import csv
import classes

def generate_grid(locations, filename = 'grid.txt'):
    """
    Function which generated a txt file with the locations specified
    IN:
        locations (list[Location]) : list of locations to be printed in the file
        filename (string) : filename to save
    OUT:
        generates file with name filename with the given locations and saves to the disk
    """
    header = []
    grid = []
    for location in locations:
        current_location = []
        for key, value in location.dimensions.items():
            if key.should_print:
                if len(grid) == 0:
                    header.append(key.name)
                current_location.append(value)
        for key, value in location.properties.items():
            if len(grid) == 0:
                header.append(key)
            current_location.append(value)
        grid.append(current_location)
    DELIMITER = ', '
    np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments = '')

#copied from stack overflow
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', autosize = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        autosize    - Optional  : automatically resize the length of the progress bar to the terminal window (Bool)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % (prefix, fill, percent, suffix)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s' % styling.replace(fill, bar), end = '\r')
    # Print New Line on Complete
    if iteration >= total:
        print()

def print_samples(samples, filename, mode):
    """
    Function that prints all the hits to a file
    IN:
        samples(list(Location)): All the samples that need to be printed
        filename (String) : The filename that will be saved
    """
    with open(filename, mode) as file:
        for sample in samples:
            current_dict = {}
            for dimension in sorted(sample.dimensions.keys(), key = lambda d: d.name):
                current_dict[dimension.name] = sample.dimensions[dimension]
            for prop in sorted(sample.properties.keys()):
                current_dict[prop] = sample.properties[prop]
            writer = csv.DictWriter(file, current_dict.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(current_dict)

def read_samples(filename, dimensions, only_hits = False):
    """
    Function that reads samples from a given file
    """
    with open(filename, newline = '') as file:
        samples = csv.DictReader(file)
        dimensions_hash = dict()
        for dimension in dimensions:
            dimensions_hash[dimension.name] = dimension
        locations = []
        for sample in samples:
            if only_hits and int(sample['is_hit']) == 0:
                continue
            sample.update((k, float(v)) for k, v in sample.items())
            locations.append(classes.Location.create_location(dimensions_hash, sample))
        return locations

def generate_slurm_file(command, batch_num, output_folder):
    slurm_folder = get_or_create_folder(output_folder, 'slurms')
    log_folder = get_or_create_folder(output_folder, 'logs')
    slurm_file = slurm_folder + "/slurm_" + str(batch_num) + ".sh"
    log_file = log_folder + "/log_" + str(batch_num) + ".txt"
    writer = open(slurm_file, 'w')
    writer.write("#!/bin/bash\n")
    writer.write("#SBATCH --output=output.out\n")
    writer.write(command + " > " + log_file + " \n")
    writer.close()
    return slurm_file

def run_code(command, batch_num, output_folder, debug = False, run_on_helios = True):
    """
    Function that runs the command specified on the command shell.
    IN:
        command list(String): A list of commands to be triggered along with the options
        batch_num (int) : The current batch number
    OUT:
        subprocess : An instance of subprocess created after running the command
    """
    if command != None:
        if not debug:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        else:
            stdout = stderr = None
        command_to_run = " ".join(str(v) for v in command)
        if run_on_helios:
            slurm_file = generate_slurm_file(" ".join(str(v) for v in command), batch_num, output_folder)
            command_to_run = "sbatch -W -Q " + slurm_file
        else:
            log_folder = get_or_create_folder(output_folder, 'logs')
            log_file = log_folder + "/log_" + str(batch_num) + ".txt"
            command_to_run = command_to_run + " > " + log_file
        process = subprocess.Popen(command_to_run, shell = True, stdout = stdout, stderr = stderr)
        return process

def get_slurm_output(output_folder, batch_num):
    try:
        log_folder = os.path.join(output_folder, 'logs')
        log_file = log_folder + "/log_" + str(batch_num) + ".txt"
        with open(log_file) as f:
            return f.readline()
    except:
        pass

def get_or_create_folder(path, name):
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder