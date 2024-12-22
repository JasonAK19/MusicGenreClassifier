# src/dataset.py
import torch
from torch.utils.data import Dataset
import librosa
import os
import numpy as np

class GTZANDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._get_files()

    def _get_files(self):
        files = []
        for genre in self.classes:
            genre_path = os.path.join(self.data_path, 'Data', genre)
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    files.append((os.path.join(genre_path, file), genre))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path, genre = self.files[idx]
        # Load audio file
        signal, sr = librosa.load(audio_path, duration=3.0, sr=22050)
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)
        
        # Convert to tensor
        mel_spec_db = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        label = self.class_to_idx[genre]
        
        return mel_spec_db, label