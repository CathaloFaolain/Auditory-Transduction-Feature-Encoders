#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-
Created on Thurs Aug 12:43 2024

@author: Cathal Ó Faoláin

The goal of this work is to understand how we can use predicted IHC potentials,
such as those predicted by WavIHC, introduced in the paper 
"WaveNet-based approximation of a cochlear filtering and hair cell transduction model". 
Feature encoders designed to use these predicted IHC potentials are evaluated against 
other state-of-the-art feature encoders in order to understand how discriminating they are,
and over a range of different Signal-to-Noise Ratios (SNRs).

This holds functions and datasets that are used by our models for handling the TIMIT dataset,
 such a collators and torch Datasets.
-
"""
import torch
from torch import nn
import librosa
import time
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchaudio
import pandas as pd
import numpy as np
import time
import sys
import yaml
import math
import scipy.signal as signal
from dataclasses import dataclass, field
from typing import List, Tuple
import torch.nn.functional as F

class TIMITDataset(Dataset):
    
    def __init__(self, DATASET : str ,
                 IHC: bool  =False,
                 TRANSFORMATION: torchaudio.transforms = None,
                 sr: int =10,
                 mel: bool=False,
                 whisper: bool=False,
                 fs: int =16000,
                 num_samples: int = 3*16000,
                kFold_eval: bool = False,
                kInt: int = 1,
                noise: bool=False,
                 noise_type: str ="White",
                 SNR: int = 30,):

        data_location="Data/"
        if noise:
            data_location = data_location + "Noisy/{}/{}/".format(noise_type, SNR)
            
        if kFold_eval==True:
            data_location=data_location + "kFolds/"
            metadata=data_location +"TIMIT_Metadata_Train_k{}.csv".format(kInt)
        elif DATASET.lower().strip() == "train":
            metadata=data_location +'TIMIT_Metadata_Train.csv'
        elif DATASET.lower().strip() == "validation":
            metadata=data_location +'TIMIT_Metadata_Validation.csv'
        elif DATASET.lower().strip() == "test":
            metadata=data_location +'TIMIT_Metadata_Test.csv'
        else:
            raise Exception("ERROR: Unsupported dataset selected: %s. Please try again with either Train, Test or Validation." %DATASET)
            
        #Load in the TIMIT metadata
        self.meta=pd.read_csv(metadata)
        
        self.transformation=TRANSFORMATION
    
        self.mel=mel
        self.fs=fs
        #Outputs sampling rate
        self.sr=sr
        self.ihc=IHC
        self.whisper=whisper
        #Give it an index from the TIMIT metadata
        
    def __len__(self):
        return len(self.meta)
    
    def get_entry(self, index):
        return self.meta.iloc[index]
    
    def __getitem__(self, index):
        signal, phonemes=self.process_data(index)
                    
        return signal, phonemes
    
    def process_data(self, index):
        #Get the audio and phoneme data paths
        audio_file=self.meta.iloc[index]['Audio']
        #Get the model used
        if self.whisper:
            fileend="Whisper"
        elif self.mel:
            fileend="Mel"
        elif self.sr==10:
            fileend="CPC"
        else:
            fileend="Wav2vec2"
            
        if self.ihc:
            phoneme_csv=self.meta.iloc[index]['IHC_Y_Outputs_%s' %fileend]
        else:
            phoneme_csv=self.meta.iloc[index]['Y_Outputs_%s' %fileend]
            
        #Load in the audio file and phoneme labels
        phonemes=pd.read_csv(phoneme_csv,  index_col=0).to_numpy()

        
        if self.mel==True:
            signal, sr=torchaudio.load(audio_file)
            #Transform the audio file as needed and get the phoneme labels
            signal=self.transformation(signal)[0].T
        else:
            #Otherwise get mono audio
            signal, _=librosa.load(audio_file, sr=self.fs)
            signal = signal/np.std(signal)*20e-6*10**(60/20)
            #Convert it to a tensor here
            signal=torch.from_numpy(signal)
        
        return signal.clone().detach(), torch.tensor(phonemes) 
    #Returns signal =   (num_phonemes, Mel dim), phonemes = (num_phonemes, class num)

#Parallelize, or otherwise use GPU if possible
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

#This allows for dynamic batch padding - only padding to the maximum length required for a batch
def batch_collator(batch):
    #Get the size of the batch
    BS=len(batch)
    
    #We will then need to get the lengths of the utterances in the batch and sort them into descending order
    s_lengths=[signal[0].shape[0] for signal in batch]
    s_lengths=torch.tensor(s_lengths)
    #The phonemes (these may have different lengths)
    p_lengths=[signal[1].shape[0] for signal in batch]
    p_lengths=torch.tensor(p_lengths)
    
    
    #This will give the sorted order of the utterances
    sorted_lengths, sort_idx=torch.sort(s_lengths, 0, descending=True)
    sorted_p_lengths, _=torch.sort(p_lengths, 0, descending=True)
    
    #Change them to shape data=(B, L, Mel_dim), target=(B, L, class_num)
    data=torch.zeros(BS, int(sorted_lengths[0]), 80)    
    target=torch.zeros(BS, int(sorted_p_lengths[0]), 39)
    
    collate_idx=0
    
    #Go through all the utterances in the batch in descending sorted order 
    for idx in sort_idx:
        #Get the desired input and output for this example.
        x, y=batch[int(idx)]
        
        #collate the inputs with the previous inputs
        data[collate_idx,: x.size()[0],:]=x
        
        #Add padded values to the end of the output
        target[collate_idx, :y.size()[0], :]= y
        
        #Increment collate index
        collate_idx+=1

    
    #Change the sorted lengths to definitely be integers prior to padding
    sorted_lengths = [int(l) for l in sorted_lengths]
    sorted_p_lengths = [int(p) for p in sorted_p_lengths]

    #Pack those fuckers together like sardines
    data = torch.nn.utils.rnn.pack_padded_sequence(data,
                                                   list(sorted_lengths),
                                                   batch_first=True)
    target = torch.nn.utils.rnn.pack_padded_sequence(target,
                                                   list(sorted_p_lengths),
                                                   batch_first=True)


    #Return the now collated data and target
    return [data, target]

#This allows for dynamic batch padding - only padding to the maximum length required for a batch
def batch_signal_collator(batch):
    #Get the size of the batch
    BS=len(batch)
    
    #We will then need to get the lengths of the utterances in the batch and sort them into descending order
    #The utterances
    s_lengths=[signal[0].shape[0] for signal in batch]
    s_lengths=torch.tensor(s_lengths)
    #The phonemes (these have different lengths)
    p_lengths=[signal[1].shape[0] for signal in batch]
    p_lengths=torch.tensor(p_lengths)
    
    #This will give the sorted order of the utterances
    sorted_lengths, sort_idx=torch.sort(s_lengths, 0, descending=True)
    sorted_p_lengths, _ = torch.sort(p_lengths, 0, descending=True)
    
    #Change them to shape data=(B, C, signal), target=(B, class_num, L)
    data=torch.zeros(BS, int(sorted_lengths[0]), 1)    
    target=torch.zeros(BS, int(sorted_p_lengths[0]), 39)
    
    collate_idx=0
    
    #Go through all the utterances in the batch in descending sorted order 
    for idx in sort_idx:
        #Get the desired input and output for this example The output should not match up 
        #- it should be able to be shaped differently, as there is 1 output for every 
        # 10ms of speech
        x, y=batch[int(idx)]
        
        
        #Flatten x into 1 dimension - we are currently capable of this because 
        #it should be a mono audio signal
        x=x.unsqueeze(dim=1)
        
        #collate the inputs with the previous inputs
        data[collate_idx, : x.size()[0], : ]=x
        
        #Add padded values to the end of the output
        target[collate_idx, :y.size()[0], : ]= y
        
        #Increment collate index
        collate_idx+=1

    
    #Change the sorted lengths to definitely be integers prior to padding
    sorted_lengths = [int(l) for l in sorted_lengths]
    sorted_p_lengths=[int(p) for p in sorted_p_lengths]

    #Pack those fuckers together like sardines
    data = torch.nn.utils.rnn.pack_padded_sequence(data,
                                                   list(sorted_lengths),
                                                   batch_first=True)
    target = torch.nn.utils.rnn.pack_padded_sequence(target,
                                                   list(sorted_p_lengths),
                                                   batch_first=True)


    #Return the now collated data and target
    return [data, target]