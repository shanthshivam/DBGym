"""
dataset.py
This module contains dataset function.
"""

import os
import requests
import zipfile
import shutil
from dbgym.db import DataBase, Tabular
from dbgym.db2pyg import DB2PyG
from yacs.config import CfgNode


def download_dataset(url, folder):
    """
    Downloads and extracts a ZIP file from the given URL to the specified folder,
    and moves the subfolders within the extracted folder to the parent folder.

    Args:
        url (str): The URL of the ZIP file to download.
        folder (str): The path to the folder where the ZIP file will be saved and extracted.

    Returns:
        None
    """
    # Download the ZIP file
    response = requests.get(url)
    if response.status_code == 200:
        # Construct the path to save the ZIP file
        zip_path = os.path.join(folder, 'dataset.zip')

        # Save the ZIP content to a local file
        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

        # Remove the saved ZIP file
        os.remove(zip_path)

        print('ZIP file downloaded and extracted to the specified folder.')

        # Get the parent folder of folder_A
        parent_folder = folder
        folder_A = os.path.join(folder, "RDBench-Dataset-master")

        # Get all subfolders within folder_A
        subfolders = [f.path for f in os.scandir(folder_A) if f.is_dir()]

        # Move the subfolders to the parent folder
        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)  # Name of the subfolder
            new_location = os.path.join(parent_folder, subfolder_name)  # New path of the subfolder
            shutil.move(subfolder, new_location)

    else:
        print('Failed to download the ZIP file. Please check the URL.')



def create_dataset(cfg: CfgNode):
    '''
    The dataset function, get dataset

    Args:
    - cfg: The configuration

    Return:
    - dataset: Tabular, DB2PyG or others
    '''

    data_dir = cfg.dataset.data_dir
    path = os.path.join(data_dir)

    # Download dataset if it doesn't exist
    if not os.path.exists(path):
        link = cfg.dataset.url 
        print(f"Downloading dataset from {link}...")
        download_dataset(link, path)
        print("Dataset downloaded successfully.")

    if cfg.dataset.type == 'single':
        tb = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        tb.load_csv()
        cfg.model.output_dim = tb.output
        return tb
    if cfg.dataset.type == 'join':
        tb = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        tb.load_join()
        cfg.model.output_dim = tb.output
        return tb
    if cfg.dataset.type == 'graph':
        db = DataBase(path)
        db.load()
        db.prepare_encoder()
        data = DB2PyG(db, cfg.dataset.file, cfg.dataset.column)
        cfg.model.output_dim = data.output
        return data
    raise ValueError(f"Model not supported: {cfg.model}")
