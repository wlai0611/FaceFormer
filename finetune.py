from faceformer import Faceformer
from transformers import TRANSFORMERS_CACHE
from pathlib import Path
import torch
import os
import argparse
from custom_data_loader import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Fine tuning Faceformer')
    parser.add_argument("--model_name", type=str, default="vocaset")
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()   

    device = 'cpu'
    cache_dir   = Path(TRANSFORMERS_CACHE).parent
    wav2vec_dir = cache_dir/"hub"/"models--facebook--wav2vec2-base-960h"
    wav2vec_dir= wav2vec_dir/"snapshots"/"22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

    #build model
    model = Faceformer(args,wav2vec_dir)
    model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),map_location=torch.device(args.device)))
    model = model.to(torch.device(args.device))
    model.eval()

    avg_template = np.load("avg_subject_template.npy").reshape(-1)
    root = Path("../caer/videos/CAER")
    folders = list(root.glob("*/*/processed/*"))
    dataset = CustomDataset(folders,avg_template)
    loader  = DataLoader(dataset,batch_size=1,shuffle=True)
    criterion = nn.MSELoss()
    for i, (audio, vertice, template, one_hot, folder_name) in enumerate(loader):
      loss = model(audio, template,  vertice, one_hot, criterion, teacher_forcing=True)
      print(loss)

    print()