# src/inference.py
import torch
import librosa
import numpy as np
import os
import argparse
from model import AudioCNN
import matplotlib.pyplot as plt
from utils import find_dataset

def predict_genre(audio_path=None, model_path='best_model.pth', show_plot=True):
    print(f"Checking paths...")
    print(f"Audio path: {os.path.abspath(audio_path)}")
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Audio exists: {os.path.exists(audio_path)}")
    print(f"Model exists: {os.path.exists(model_path)}")
    if audio_path is None:
        dataset_path = find_dataset()
        if dataset_path is None:
            raise FileNotFoundError("GTZAN dataset not found. Please run dataset_download.py first")
        audio_path = os.path.join(dataset_path, 'jazz', 'jazz.00001.wav')

    # Validate paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load trained model
    try:
        model = AudioCNN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

    # Load and preprocess audio
    try:
        signal, sr = librosa.load(audio_path, duration=3.0, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    except Exception as e:
        raise RuntimeError(f"Error processing audio: {str(e)}")

    # Convert to tensor
    input_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.bar(classes, probabilities.numpy())
        plt.xticks(rotation=45)
        plt.title('Genre Probabilities')
        plt.ylabel('Probability')
        plt.xlabel('Genre')
        plt.tight_layout()
        plt.show()
    
    predicted_genre = classes[predicted.item()]
    confidence = float(probabilities[predicted].item())
    
    return predicted_genre, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict music genre from audio file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model file')
    parser.add_argument('--no-plot', action='store_true', help='Disable probability plot')
    args = parser.parse_args()

    try:
        print(f"Starting prediction...")
        genre, confidence = predict_genre(
            audio_path=args.audio,
            model_path=args.model,
            show_plot=not args.no_plot
        )
        print(f"Predicted genre: {genre}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. You've run dataset_download.py to download the GTZAN dataset")
        print("2. You've trained the model using train.py")
        print("3. The model file (best_model.pth) exists in the current directory")

if __name__ == "__main__":
    main()