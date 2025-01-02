import os

def find_dataset():
    """Find GTZAN dataset in common locations"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        os.path.join(base_dir, 'data', 'genres_original'),
        os.path.join(base_dir, 'Data', 'genres_original'),
        os.path.join(base_dir, 'data', 'Data', 'genres_original'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None