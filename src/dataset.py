# In src/dataset.py
import torch
from torch.utils.data import Dataset
import librosa
import os
import numpy as np

class GTZANDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
                    'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        possible_paths = [
            os.path.join(data_path, 'Data', 'genres_original'),
            os.path.join(data_path, 'genres_original'),
            os.path.join(data_path, 'Data'),
            data_path
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and any(os.path.exists(os.path.join(path, genre)) for genre in self.classes):
                self.data_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find valid genre folders in any expected locations")
                
        self.files = self._get_files()
        
        # preprocessing parameters
        self.sample_rate = 22050
        self.duration = 3  
        self.n_mels = 128
        self.time_steps = 130
        self.n_fft = 2048
        self.hop_length = 512

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
       try:
            file_path, genre = self.files[idx]
            
            audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            if len(audio) < self.sample_rate * self.duration:
                audio = np.pad(audio, (0, self.sample_rate * self.duration - len(audio)))
            else:
                audio = audio[:self.sample_rate * self.duration]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if mel_spec_db.shape[1] < self.time_steps:
                pad_width = ((0, 0), (0, self.time_steps - mel_spec_db.shape[1]))
                mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :self.time_steps]
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Convert to tensor
            mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            label = torch.tensor(self.class_to_idx[genre])
            
            return mel_spec_tensor, label
            
       except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Return a zero tensor with correct dimensions and -1 label
            return torch.zeros((1, self.n_mels, self.time_steps)), torch.tensor(-1)
    def _get_files(self):
        files = []
        for genre in self.classes:
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.exists(genre_path):
                print(f"Warning: Genre directory not found: {genre_path}")
                continue
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    files.append((os.path.join(genre_path, file), genre))
        if not files:
            raise RuntimeError("No .wav files found in the data directory")
        return files
        
if __name__ == "__main__":
    # Test data loading
    try:
        dataset = GTZANDataset("data")
        print(f"Successfully loaded {len(dataset)} audio files")
        print(f"Data path: {dataset.data_path}")
        print(f"Sample paths:")
        for i in range(min(3, len(dataset))):
            print(f"  {dataset.files[i][0]}")
    except Exception as e:
        print(f"Error loading dataset: {e}")