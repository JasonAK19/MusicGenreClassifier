# dataset_download.py
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys

def download_gtzan():
    try:
        print("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory
        data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_path, exist_ok=True)
        
        print("Downloading GTZAN dataset...")
        api.dataset_download_files(
            'andradaolteanu/gtzan-dataset-music-genre-classification',
            path=data_path,
            unzip=True,
            quiet=False
        )
        
        print(f"Dataset downloaded successfully to: {data_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_gtzan()