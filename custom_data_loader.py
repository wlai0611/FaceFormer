from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from torch import nn
import librosa
from transformers import Wav2Vec2Processor
from pathlib import Path
def preprocess_audio_data(folders):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    for folder in folders:
      if folder.is_dir():
        #copied from FaceFormer
        wav_path = next(folder.glob("*.wav"))
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        np.save(folder/'audio.npy',input_values)

class CustomDataset(Dataset):
  """Root folder with multiple subfolders each with blendshape,WAV of each video"""
  def __init__(self, folders, template, n_subjects=8):
    #n_subjects needs to be 8 to match the trainset of VOCASET on which FaceFormer trained
    self.folders = folders
    self.template = torch.FloatTensor(template)
    #the average of the standard basis in 8D to represent average subject
    self.subject_embedding = torch.ones(n_subjects)/n_subjects
    self.subject_embedding.to(torch.float)
  def __len__(self):
    return len(self.folders)
  def __getitem__(self,index):
    '''
    Inside each folder must have NPZ containing "verts" of shape (T,5023,3)
    each of the T frames should be blendshapes sampled at 30 FPS 
    (VOCASET recorded at 60 fps, FaceFormer downsampled to 30 FPS for training, we match)
    Also each folder must contain NPY of a single 1D vector of speech sampled at 16kHz 
    (following FaceFormer) using pretrained Wav2Vec2
    '''
    folder = self.folders[index]
    face_data = np.load(folder/"blendshapes.npz")
    audio = torch.FloatTensor(np.load(folder/"audio.npy"))
    verts = torch.FloatTensor(face_data['verts'])
    verts = verts.flatten(start_dim=1)
    return audio, verts, self.template, self.subject_embedding, folder.name
    #return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(self.template), torch.FloatTensor(self.subject_embedding)
if __name__ == "__main__":
  avg_template = np.load("avg_subject_template.npy").reshape(-1)
  root = Path("../caer/videos/CAER")
  folders = list(root.glob("*/*/processed/*"))
  preprocess_audio_data(folders)
  dataset = CustomDataset(folders,avg_template)
  loader  = DataLoader(dataset,batch_size=1,shuffle=True)
  for i, (audio, vertice, template, one_hot, folder_name) in enumerate(loader):
    pass 
  criterion = nn.MSELoss()
  #loss = model(audio, template,  vertice, one_hot, criterion, teacher_forcing=True)