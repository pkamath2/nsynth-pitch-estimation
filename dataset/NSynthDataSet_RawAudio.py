import librosa
from librosa.core import audio, magphase
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json
import util

class NSynthDataSet_RawAudio(Dataset):
    def __init__(self, type):

        config = util.get_config()
        self.instrument = config['instrument']
        self.lower_pitch_limit = int(config['lower_pitch_limit'])
        self.upper_pitch_limit = int(config['upper_pitch_limit'])
        self.sample_length = int(config['sample_length'])
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]
        self.sr = int(config['sample_rate'])
        self.base_data_dir = config['base_data_dir']

        self.meta_data_file = os.path.join(self.base_data_dir, config[type]['label_dir'])
        self.audio_dir = os.path.join(self.base_data_dir, config[type]['audio_dir'])
        
        with open(self.meta_data_file) as f:
            params = json.load(f)
            self.nsynth_meta_df = pd.DataFrame.from_dict(params)
            self.nsynth_meta_df = self.nsynth_meta_df.transpose()
            self.nsynth_meta_df = self.nsynth_meta_df[self.nsynth_meta_df['instrument_family_str'] == self.instrument]
            self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['pitch'] >= self.lower_pitch_limit) \
                                                      & (self.nsynth_meta_df['pitch'] < self.upper_pitch_limit)]
            
            self.nsynth_meta_df['part'] = 1
            nsynth_meta_df_2 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_2['part'] = 2
            nsynth_meta_df_2.index = nsynth_meta_df_2.index + '-2'
            nsynth_meta_df_3 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_3['part'] = 3
            nsynth_meta_df_3.index = nsynth_meta_df_3.index + '-3'
            nsynth_meta_df_4 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_4['part'] = 4
            nsynth_meta_df_4.index = nsynth_meta_df_4.index + '-4'
            self.nsynth_meta_df = pd.concat([self.nsynth_meta_df, nsynth_meta_df_2, nsynth_meta_df_3, nsynth_meta_df_4])
        
    def __len__(self):
        return self.nsynth_meta_df.shape[0]
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx): #Ensure the elements are always scalars
            idx = idx.tolist()
        audio_file_name = self.nsynth_meta_df.iloc[idx].note_str + '.wav'
        audio_pitch = self.nsynth_meta_df.iloc[idx].pitch
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr)
        
        mult = 0.25 + ((self.nsynth_meta_df.iloc[idx].part - 1) * 0.5)
        start_location = int(16000 * mult)
        
        audio_data = audio_data[start_location:start_location + self.sample_length]
        audio_data_stft = librosa.stft(audio_data, n_fft=(self.sample_length-1)*2)
        audio_data = librosa.mu_compress(audio_data, quantize=False)
        audio_data = librosa.util.normalize(audio_data)
        new_sample = np.concatenate([np.reshape(audio_data, (self.sample_length, 1)) * 2, np.abs(audio_data_stft),np.angle(audio_data_stft)], axis=1)
        return new_sample, self.classes.index(audio_pitch)
