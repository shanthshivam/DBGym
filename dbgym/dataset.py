"""
dataset.py
This module contains dataset function.
"""

import os
import zipfile
import shutil
import requests
from tqdm import tqdm
from dbgym.db import DataBase, Tabular
from dbgym.db2pyg import DB2PyG
from yacs.config import CfgNode


def download_dataset(url, folder):
    """
    Downloads and extracts a ZIP file from the given URL to the specified folder,
    and moves the subfolders within the extracted folder to the parent folder.

    Args:
    - url (str): The URL of the ZIP file to download.
    - folder (str): The path to the folder where the ZIP file will be saved and extracted.
    """

    # Download the ZIP file
    response = requests.get(url, stream=True, timeout=15)
    if response.status_code == 200:
        # Construct the path to save the ZIP file
        zip_path = os.path.join(folder, 'dataset.zip')

        fmat = "{l_bar}{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}, {rate_fmt}{postfix}"
        progress_bar = tqdm(total=95025233, unit="B", unit_scale=True, bar_format=fmat)
        downloaded_size = 0

        # Save the ZIP content to a local file
        with open(zip_path, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                downloaded_size += len(data)
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

        # Remove the saved ZIP file
        os.remove(zip_path)

        print('ZIP file downloaded and extracted to the specified folder.')

        # Get the parent folder of folder_a
        parent_folder = folder
        folder_a = os.path.join(folder, "RDBench-Dataset-master")

        # Get all subfolders within folder_a
        subfolders = [f.path for f in os.scandir(folder_a) if f.is_dir()]

        # Move the subfolders to the parent folder
        for subfolder in subfolders:
            # Name of the subfolder
            subfolder_name = os.path.basename(subfolder)
            # New path of the subfolder
            new_location = os.path.join(parent_folder, subfolder_name)
            shutil.move(subfolder, new_location)

        old_name = os.path.join(folder, "RDBench-Dataset-master")
        new_name = os.path.join(folder, "info")
        os.rename(old_name, new_name)
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

    data_dir = cfg.dataset.dir
    path = os.path.join(data_dir, cfg.dataset.name)

    # Download dataset if it doesn't exist
    if cfg.dataset.name != 'example' and not os.path.exists(path):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        link = 'https://github.com/YiYang-github/RDBench-Dataset/archive/refs/heads/master.zip'
        print(f"Downloading dataset from {link}...")
        download_dataset(link, data_dir)
        print("Dataset downloaded successfully.")

    if cfg.dataset.type == 'tabular':
        tabular = Tabular(path, cfg.dataset.file, cfg.dataset.column)
        if cfg.dataset.format == 'single':
            tabular.load_csv()
        if cfg.dataset.format == 'join':
            tabular.load_join()
        cfg.model.output_dim = tabular.output
        return tabular

    if cfg.dataset.type == 'graph':
        database = DataBase(path)
        database.load()
        database.prepare_encoder()
        graph = DB2PyG(database, cfg.dataset.file, cfg.dataset.column)
        cfg.model.output_dim = graph.output
        return graph

    raise ValueError(f"Dataset type not supported: {cfg.dataset.type}")
