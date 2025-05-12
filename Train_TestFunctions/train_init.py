"""
-
Created on Wednes 14th Aug 2024

@author: Cathal Ó Faoláin

The goal of this work is to understand how we can use predicted IHC potentials,
such as those predicted by WavIHC, introduced in the paper 
"WaveNet-based approximation of a cochlear filtering and hair cell transduction model". 
Feature encoders designed to use these predicted IHC potentials are evaluated against 
other state-of-the-art feature encoders in order to understand how discriminating they are,
and over a range of different Signal-to-Noise Ratios (SNRs).

This holds functions designed to help initialise our training and testing instances.
As we are relying on new features, we need to train our own models, and test/evaluate them in interesting ways
owing to the different types of noise - either steady or amplitude modulated, and the subtypes therein. The
novel use of predicted IHC potentials also relies on models that have to be trained from scratch here.
-
"""
import torch
from torch import nn
import librosa
import time
import pathlib as Path
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchaudio
import pandas as pd
import numpy as np
import time
import sys
import yaml
import math
import os
import scipy.signal as signal
from dataclasses import dataclass, field
from typing import List, Tuple
import torch.nn.functional as F
#import tensorboard_logger
#from tensorboard_logger import log_value
import pickle
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torch.distributed import init_process_group, destroy_process_group
from TIMIT_utils import TIMIT_utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
import torch.distributed as dist
from torchsummary import summary

from Encoders import FeatureEncoders

class EarlyStopper:
    """
    This class allows for early stopping if validation loss begins to increase - i.e. there is 
    evidence that the model has begun to overfit. 
    Code taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    Args:
        - patience: the number of times in a row we allow for the validation loss to drop below zero before stopping
        - min_delta: the range of validation loss we allow to vary before concluding that the model is getting worse
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False

def load_checkpoint(model, optimizer, checkpoint):
    """
    This helper function allows us to load in a checkpoint, of both the model and optimizer. This faciliates
    either further training or testing of a previous model. 
    Args:
        - model: model object of the same type we want to load in a checkpoint of
        - optimizer: the optimizer type used in training before
        - checkpoint: the location where the model checkpoint is stored
    Returns:
        - model: trained and loaded in torch.nn.Module
        - optimizer: loaded in torch.optim.Optimizer
    """
    print("Loading model checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return model, optimizer
    
def prepare_dataloader(dataset : Dataset, batch_size: int, batch_collator, distributed : bool=True):
        """
        This sets up a dataloader with the arguments required for distributed model training or serial training.
        Args: 
            - dataset: The Torch Dataset the dataloader will rely on
            - batch_size: the number of samples per batch
            - batch_collator: the collator function to be used to variably pad batches
            - distributed: return a dataloader set up for distributed or not. Default true
        Returns:
            - torch.utils.data.DataLoader
        """
        if distributed:
            return DataLoader(dataset,
                          batch_size=batch_size,
                          pin_memory= True,
                          shuffle =False,
                          sampler = DistributedSampler(dataset),
                          collate_fn=batch_collator
                         )
        else:
             return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=batch_collator
                         )
            
def K_fold_datasets(kInt: int, TRANSFORMATION, mel, whisper, sr, IHC, noise, noise_type, SNR):
    """
        This sets ups the k_fold datasets, splitting them into the validation and training sets based on the kInt value 
        Args: 
            - kInt: The k-Fold that will be used as the validation set
            - TRANSFORMATION: the transformation operator (if any) to be done to the signal at the dataset loading stage
            - mel: Whether or not to perform mel spectrogram transformation on the signal
            - sr: the sample rate to return
            - IHC: whether or not to return the dataset ready for an IHC based model (removes the first 2047 samples)
            - noise: whether or not to select a noisy dataset
            - noise_type : if noise is selected, what kind of noise to use
            - SNR : what SNR value to use, if noisy 
        Returns:
            - train_k : TIMIT_utils.Dataset
            - valid_k: TIMIT_utils.Dataset
        """
    #Get the datasets
    k1=TIMIT_utils.TIMITDataset("Data/kFolds/TIMIT_Metadata_Train_k1.csv", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, kFold_eval=True, kInt=1, noise=noise, noise_type=noise_type, SNR=SNR)
    k2=TIMIT_utils.TIMITDataset("Data/kFolds/TIMIT_Metadata_Train_k2.csv", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, kFold_eval=True, kInt=2, noise=noise, noise_type=noise_type, SNR=SNR)
    k3=TIMIT_utils.TIMITDataset("Data/kFolds/TIMIT_Metadata_Train_k3.csv", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, kFold_eval=True, kInt=3, noise=noise, noise_type=noise_type, SNR=SNR)
    k4=TIMIT_utils.TIMITDataset("Data/kFolds/TIMIT_Metadata_Train_k4.csv", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, kFold_eval=True, kInt=4, noise=noise, noise_type=noise_type, SNR=SNR)
    k5=TIMIT_utils.TIMITDataset("Data/kFolds/TIMIT_Metadata_Train_k5.csv", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, kFold_eval=True, kInt=5, noise=noise, noise_type=noise_type, SNR=SNR)

    #Combine the list of k folds
    kList=[k1, k2, k3, k4, k5]

    #Pop off the validation set
    valid_k=kList.pop(kInt)
    print(">> Returning TIMIT datasets, using k-fold %d as the validation set" %(kInt+1))

    #Concat the other datasets to combine them
    train_k=ConcatDataset(kList)
    
    return train_k, valid_k
    
def model_selector(model_name: str):
    """
    This function returns the model associated with a given input string.
    Args: 
        - model_name : Name of the desired model
    Returns:
        - model : torch.nn.Module
    """
    if model_name.lower().find("ihc_wav2vec2_80") != -1:
        model= FeatureEncoders.IHC_Wav2vec2(nHiddenUnits=80,
            dropout=0.0,
            mode="default",
            conv_bias=False)
        print("> Initialising model: IHC_Wav2Vec2_80")
        return model
    elif model_name.lower().find("ihc_wav2vec2") != -1:
        print("> Initialising model: IHC_Wav2Vec2")
        return FeatureEncoders.IHC_Wav2vec2()
    elif model_name.lower().find("ihc_cpc_80") != -1:
        print("> Initialising model: IHC_CPC_80")
        return FeatureEncoders.IHC_Cpc(nHiddenUnits=80)
    elif model_name.lower().find("ihc_cpc") != -1:
        print("> Initialising model: IHC_CPC")
        return FeatureEncoders.IHC_Cpc()
    elif model_name.lower().find("ihc_extract_512") !=-1:
        print("> Initialising model: IHC_Extract_512")
        return FeatureEncoders.IHC_Extract(nHiddenUnits=512)
    elif model_name.lower().find("ihc_extract_2.0") !=-1:
        print("> Initialising model: IHC_Extract_2.0")
        return FeatureEncoders.IHC_Extract_2()
    elif model_name.lower().find("ihc_extract_3.0") !=-1:
        print("> Initialising model: IHC_Extract_3.0")
        return FeatureEncoders.IHC_Extract_3()
    elif model_name.lower().find("ihc_extract") !=-1:
        print("> Initialising model: IHC_Extract")
        return FeatureEncoders.IHC_Extract()
    elif model_name.lower().find("sig_extract_512") !=-1:
        print("> Initialising model: SIG_Extract_512")
        return FeatureEncoders.SIG_Extract(nHiddenUnits=512)
    elif model_name.lower().find("sig_extract_2.0") !=-1:
        print("> Initialising model: SIG_Extract_2.0")
        return FeatureEncoders.SIG_Extract_2()
    elif model_name.lower().find("sig_extract_3.0") !=-1:
        print("> Initialising model: SIG_Extract_3.0")
        return FeatureEncoders.SIG_Extract_3()
    elif model_name.lower().find("sig_extract") !=-1:
        print("> Initialising model: SIG_Extract")
        return FeatureEncoders.SIG_Extract()
    elif model_name.lower().find("cpc_80") != -1:
        print("> Initialising model: CPC_80")
        return FeatureEncoders.CpcEncoder(nHiddenUnits=80)
    elif model_name.lower().find("cpc") != -1:
        print("> Initialising model: CPC")
        return FeatureEncoders.CpcEncoder()
    elif model_name.lower().find("wav2vec2_80") != -1:
        model= FeatureEncoders.Wav2vec2Encoder(nHiddenUnits=80,
            dropout=0.0,
            mode="default",
            conv_bias=False)
        print("> Initialising model: Wav2Vec2.0_80")
        return model
    elif model_name.lower().find("wav2vec2") != -1:
        model= FeatureEncoders.Wav2vec2Encoder(nHiddenUnits=512,
            dropout=0.0,
            mode="default",
            conv_bias=False)
        print("> Initialising model: Wav2Vec2.0")
        return model
    #elif model_name.lower().find("melSimple_1") != -1:
        #mlp_config=
        #return FeatureEncoders.MelSimple(80)
    elif model_name.lower().find("melsimple_mlp") != -1:
        print("> Initialising model: MelSimple_MLP")
        return FeatureEncoders.MelSimple(80, layers=True)
    elif model_name.lower().find("mel") != -1:
        print("> Initialising model: MelSimple")
        return FeatureEncoders.MelSimple(80)
    elif model_name.lower().find("whisper_80") != -1:
        print("> Initialising model: Whisper_80")
        return FeatureEncoders.WhisperEncoder(n_state=80)
    elif model_name.lower().find("whisper") != -1:
        print("> Initialising model: Whisper")
        return FeatureEncoders.WhisperEncoder()
    else:
        raise Exception("Unrecognised model name, please try again")

def load_train_settings(model: torch.nn.Module, distributed: bool=True, world_size: int =0):
    """
    This function loads the training data settings. It selects what kind of data is used for 
    the model training by returning the arguments used for the model's dataset.
    Args:
        - model: torch.nn.Module
    Return:
        - signal: bool,
        - IHC: bool,
        - sr : int,
        - mel : bool,
        - TRANSFORMATION : either MelSpectrogram with settings, or None 
        - batch_large: bool,
    """
        #Set the model settings
    if(model.toString().lower().find("ihc_extract_2")!=-1):
        signal=True
        IHC=True
        sr=20
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    if(model.toString().lower().find("ihc_extract_3")!=-1):
        signal=True
        IHC=True
        sr=20
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("ihc_extract")!=-1):
        signal=True
        IHC=True
        sr=10
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("extract_2")!=-1):
        signal=True
        IHC=False
        sr=20
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("extract_3")!=-1):
        signal=True
        IHC=False
        sr=20
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("extract")!=-1):
        signal=True
        IHC=False
        sr=10
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("ihc_wav2vec2")!=-1):
        signal=True
        sr=20
        IHC=True
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    elif(model.toString().lower().find("ihc_cpc")!=-1):
        signal=True
        IHC=True
        sr=10
        mel=False
        whisper=False
        TRANSFORMATION=None
        batch_large=False
    if(model.toString().lower().find("cpc_encoder")!=-1):
        signal=True
        sr=10
        mel=False
        whisper=False
        TRANSFORMATION=None
        IHC=False
        batch_large=False
    elif(model.toString().lower().find("wav2vec2.0_encoder")!=-1):
        signal=True
        sr=20
        mel=False
        whisper=False
        TRANSFORMATION=None
        IHC=False
        batch_large=False
    elif(model.toString().lower().find("melsimple")!=-1):
        signal=False
        mel=True
        sr=10
        IHC=False
        whisper=False
        SAMPLE_RATE=16000
        TRANSFORMATION=torchaudio.transforms.MelSpectrogram(
        sample_rate= SAMPLE_RATE,
        n_fft=int(math.ceil(0.025*SAMPLE_RATE)),
        hop_length=int(math.ceil(0.01*SAMPLE_RATE)),
        n_mels=80 )
        batch_large=True
    elif(model.toString().lower().find("whisper")!=-1):
        signal=False
        mel=True
        sr=10
        whisper=True
        IHC=False
        SAMPLE_RATE=16000
        TRANSFORMATION=torchaudio.transforms.MelSpectrogram(
        sample_rate= SAMPLE_RATE,
        n_fft=int(math.ceil(0.025*SAMPLE_RATE)),
        hop_length=int(math.ceil(0.01*SAMPLE_RATE)),
        n_mels=80 )
        batch_large=True
    
    return signal, IHC, sr, mel, whisper, TRANSFORMATION, batch_large

def divisble_batch(batch_large: bool, numerator: int, denominator: int):
    """
    This function calculates a batch size for the model, based on the test dataset length.
    For distributed training, these have to be divisble by the number of GPUs to prevent issues.
    Args:
        - batch_large: bool indicating if the model can handle a large batch size
        - numerator: the numerator, corresponding to the test dataset length
        - denominator: the denominator, corresponding to world size.
    Returns:
        - BATCH_SIZE: int
    """
    if batch_large:
        for BATCH_SIZE in range(15, 35):
            if numerator % denominator == 0:
                return BATCH_SIZE

    #Ends up here if either the batch size is small, or a suitable larger batch size wasn't found
    for BATCH_SIZE in range(2, 10):
        if numerator % denominator == 0:
            return BATCH_SIZE

    #Otherwise 1 definitely works as a batch size
    return 1

def all_models_parameters():
    #This is is a list of all the models, and a dict to hold the number of parameters they have
    models=["Wav2vec2", "CPC", "MelSimple", "MelSimple_MLP", "Whisper",  "Whisper_80", "Wav2vec2_80", "CPC_80","IHC_Cpc", "IHC_Cpc_80",\
            "IHC_Wav2vec2_80", "IHC_Wav2vec2", "IHC_Extract", "IHC_Extract_512", "IHC_Extract_2.0", "IHC_Extract_3.0",\
           "SIG_Extract", "SIG_Extract_512", "SIG_Extract_2.0", "SIG_Extract_3.0"]
    trainable_dict={}
    model_dict={}
    
    for model in models:
        #Get the model and the number of parameters
        module=model_selector(model)
        #Get the number of trainable parameters
        parameters=sum(p.numel() for p in module.parameters())
        t_params=sum(p.numel() if p.requires_grad == True else 0 for p in module.parameters())
    

        #print(module)
        model_dict[model]=parameters
        trainable_dict[model]=t_params
        
    return model_dict, trainable_dict

def load_training_objects(model_name: str, learning_rate: int, test: bool=False, distributed: bool=True, world_size: int = 0, Kfold_eval: bool = False, kInt: int= None, noise: bool=False, noise_type: str=None, SNR: int=30 ):
    """
    This function loads in the objects that are required for training - the model, optimizer, and
    specific settings and dataloaders. This was designed to reduce overhead at initialisation in 
    particular for distributed training. If we're testing the model rather than training, only
    one dataset is returned, the test dataset, and no scheduler.

    This is also set up for Kfold testing, to check the stability of our models. This is extremely similar to 
    regular training, with the validation set being made up of the required split.
    Args:
        - model_name: the name, or extremely similar, of the model we want to train
        - learning_rate: the learning rate of the model, also used for initialising the optimizer
        - distributed: Return the dataloaders set up for distributed or not. Default True
        - test: bool, whether or not we're loading for model testing. Defaults to false 
        - Kfold_eval: whether or not this is a kFold stability test. Defaults to false
        - KInt: If this is a kFold test, what kFold set is the validation for this run. Defaults to None
        - noise: whether or not to use a noisy dataset
        - noise_type : what kind of noisy dataset to use. Only takes effect if noise is selected
        - SNR : what Signal-to-Noise ratio dataset to use if noise is selected
    Returns:
        - model : torch.nn.Module
        - optimizer : torch.optim.Optimizer
        if test == False (default):
        - scheduler : torch.nn.lr_scheduler
        - trainloader: torch.utils.data.DataLoader. Training Dataset
        - valid_loader: torch.utils.data.DataLoader. Validation Dataset
        if test == True:
        - testloader: torch.utils.data.DataLoader. Test dataset
    """

    model=model_selector(model_name)
    
    #The optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    signal, IHC, sr, mel, whisper, TRANSFORMATION, batch_large=load_train_settings(model, distributed, world_size)

    #Create dataloaders for our model, using either the signal or mel collator
    batch_collator=  TIMIT_utils.batch_signal_collator if signal else TIMIT_utils.batch_collator
    
    #Get the dataloaders set up
    if not test:
        #The scheduler is only returned if the model is training
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

        
        if Kfold_eval:
            print("> Setting: K-Fold Training Mode")
            train_k, valid_k=K_fold_datasets(kInt, TRANSFORMATION, mel, whisper, sr, IHC, noise=noise, noise_type=noise_type, SNR=SNR)

            if world_size != 0:
            #Get the TIMITkfold size with a batchsize of required
                length=len(valid_k)
                BATCH_SIZE=divisble_batch(batch_large, length, world_size)
            else:
                BATCH_SIZE= 24 if batch_large else 4
    
            trainloader=prepare_dataloader(train_k, BATCH_SIZE, batch_collator, distributed=distributed)
            valid_loader=prepare_dataloader(valid_k, BATCH_SIZE, batch_collator, distributed=distributed)
            
            return model, optimizer, scheduler, trainloader, valid_loader
        else:
            print("> Setting: Default Training Mode")
            if world_size != 0:
            #Get the TIMITkfold size with a batchsize of required
                length=len(TIMIT_utils.TIMITDataset("validation", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, noise=noise, noise_type=noise_type, SNR=SNR))
                BATCH_SIZE=divisble_batch(batch_large, length, world_size)
            else:
                BATCH_SIZE= 24 if batch_large else 4
            
        
            trainloader=prepare_dataloader(TIMIT_utils.TIMITDataset("train", 
                                                            TRANSFORMATION=TRANSFORMATION,
                                                            mel=mel, sr=sr, whisper=whisper, IHC=IHC, noise=noise, noise_type=noise_type, SNR=SNR),
                                   BATCH_SIZE,
                                   batch_collator, distributed=distributed)

        
            valid_loader=prepare_dataloader(TIMIT_utils.TIMITDataset("validation", 
                                                            TRANSFORMATION=TRANSFORMATION,
                                                            mel=mel, whisper=whisper, sr=sr, IHC=IHC, noise_type=noise_type, SNR=SNR),
                                       BATCH_SIZE,
                                       batch_collator, distributed=distributed)
            return model, optimizer, scheduler, trainloader, valid_loader
            
    else:
        print("> Setting: Test Mode")
        if world_size != 0:
            length=len(TIMIT_utils.TIMITDataset("test", TRANSFORMATION=TRANSFORMATION, mel=mel, whisper=whisper, sr=sr, IHC=IHC, noise=noise, noise_type=noise_type, SNR=SNR))
            BATCH_SIZE=divisble_batch(batch_large, length, world_size)
        else:
            BATCH_SIZE= 24 if batch_large else 4
        
        testloader=prepare_dataloader(TIMIT_utils.TIMITDataset("test", 
                                                            TRANSFORMATION=TRANSFORMATION,
                                                            mel=mel, whisper=whisper, sr=sr, IHC=IHC, noise=noise, noise_type=noise_type, SNR=SNR),
                                   BATCH_SIZE,
                                   batch_collator, distributed=distributed)
        return model, optimizer, testloader
    

    