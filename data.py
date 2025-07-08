import glob
import os
import re
import glob
import pandas as pd
from tqdm import tqdm


DL1_PATH = '/dl1/event/telescope/parameters/LST_LSTCam'
DL2_PATH = '/dl2/event/telescope/parameters/LST_LSTCam'
# TRAINING_DIR="/Users/thomas/Work/Projets/CTA/data/LST/DL1/AllSky/20240918_v0.10.12_allsky_nsb_tuning_0.00/TrainingDataset"
# TESTING_DIR="/Users/thomas/Work/Projets/CTA/data/LST/DL2/AllSky/20240918_v0.10.12_allsky_nsb_tuning_0.00/TestingDataset/"
TRAINING_DIR = "/lapp_data/cta/Data/LST1/MC/DL1/AllSky/20240918_v0.10.12_allsky_nsb_tuning_0.00/TrainingDataset/"
TESTING_DIR = "/lapp_data/cta/Data/LST1/MC/DL2/AllSky/20240918_v0.10.12_allsky_nsb_tuning_0.00/TestingDataset/"


def get_training_dir():
    """
    Returns the path to the training directory.
    """
    return TRAINING_DIR


def get_training_files():
    """
    Returns a list of training files in the training directory.
    """
    return glob.glob(os.path.join(TRAINING_DIR, '**/*.h5'), recursive=True)


def get_testing_dir():
    """
    Returns the path to the testing directory.
    """
    return TESTING_DIR


def get_testing_files():
    """
    Returns a list of testing files in the testing directory.
    """
    return glob.glob(os.path.join(TESTING_DIR, '**/*.h5'), recursive=True)


def list_decs():
    """
    list decs dir in training directory.
    """
    gamma_decs = []
    protons_decs = []
    decs = []
    for root, dirs, files in os.walk(f"{TRAINING_DIR}/GammaDiffuse/"):
        gamma_decs.extend(dirs)
        break
    for root, dirs, files in os.walk(f"{TRAINING_DIR}/Protons/"):
        protons_decs.extend(dirs)
        break
    decs = [value for value in gamma_decs if value in protons_decs]
    return set(decs)


def read_dl1(filename, stop=None):
    """
    Lit un fichier HDF5 et retourne un DataFrame Pandas avec les données.
    """
    try:
        df = pd.read_hdf(filename, key=DL1_PATH, mode='r', stop=stop)
        print(f"Fichier lu avec succès : {filename}")
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filename}: {e}")
        return None


def read_dl2(filename, stop=None):
    """
    Lit un fichier HDF5 et retourne un DataFrame Pandas avec les données DL2.
    """
    try:
        df = pd.read_hdf(filename, key=DL2_PATH, mode='r', stop=stop)
        print(f"Fichier DL2 lu avec succès : {filename}")
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier DL2 {filename}: {e}")
        return None


def _read_training_part(part, decs=None, stop=None):
    """
    Reads a specific part of the training data (either 'GammaDiffuse' or 'Protons') and returns a DataFrame.
    """
    decs = decs if decs is not None else list_decs()
    files = []
    for dec in decs:
        dec_files = glob.glob(os.path.join(TRAINING_DIR, part, dec, '**/*.h5'), recursive=True)
        print(f"Found {len(dec_files)} files for DEC {dec} in {part}")
        if not dec_files:
            print(f"No files found for DEC {dec} in {part}. Skipping.")
            continue
        files.extend(dec_files)

    dfs = []
    for file in tqdm(files, desc=f"Reading {part} files"):
        df = read_dl1(file, stop=stop)
        if df is not None:
            dfs.append(df)
        else:
            raise ValueError(f"Failed to read file: {file}")
    return pd.concat(dfs, ignore_index=True)

def read_training_gammas(decs=None, stop=None):
    """
    Reads all gamma training files and returns a DataFrame.
    """
    return _read_training_part('GammaDiffuse', decs=decs, stop=stop)


def read_training_protons(decs=None, stop=None):
    """
    Reads all proton training files and returns a DataFrame.
    """
    return _read_training_part('Protons', decs=decs, stop=stop)


def read_training_data(decs=None, stop=None):
    """
    Reads all training data (both gammas and protons) and returns a DataFrame.
    """
    gamma_df = read_training_gammas(decs=decs, stop=stop)
    proton_df = read_training_protons(decs=decs, stop=stop)
    
    if gamma_df.empty and proton_df.empty:
        raise ValueError("No training data found for gammas or protons.")
    else:
        return pd.concat([gamma_df, proton_df], ignore_index=True)


def extract_theta_az_from_filename(filename):
    """
    Extracts theta and az values from a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        tuple: A tuple containing theta and az values as floats.
                Returns (None, None) if the values cannot be extracted.
    """
    try:
        theta_match = re.search(r'theta_([\d.]+)', filename)
        az_match = re.search(r'az_([\d.]+)', filename)

        if theta_match and az_match:
            theta = float(theta_match.group(1))
            az = float(az_match.group(1))
            return theta, az
        else:
            return None, None
    except Exception as e:
        print(f"Error extracting theta/az from filename {filename}: {e}")
        return None, None