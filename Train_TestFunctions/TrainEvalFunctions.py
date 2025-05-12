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

This holds training and testing helper functions that we can use to train and test our feature encoders.
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
import pickle
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from TIMIT_utils import TIMIT_utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
import torch.distributed as dist

#Local imports 
from Encoders import FeatureEncoders
from .Distributed_train import distributed_main, Trainer
from .train_init import model_selector, load_train_settings, load_training_objects, prepare_dataloader, load_checkpoint, all_models_parameters
from .Serial_train import serial_train, evaluate

    
#####################################################
def kFold_test(model_name:str, EPOCHS:int, learning_rate : int=0.01, distributed: bool=True):
    """
    This function is designed to test the model for stability in results, and calls on a selection of helper functions to acheive the required performance.
    This is meant to be used before the full train_epochs, and makes sure of the variable range of our models using the training data alone.
    This both trains and then immediately tests the best resulting model on the k-1 fold. Training can either be done on a single device (serial train) or 
    distributed (mp.spawn(distributed_main), based on the distributed flag and the possible detection of multiple gpus.
    Args:
        - model_name: a string name identifying the model we want to train.
        - EPOCHS: the total number of epochs we want to train our model for.
        - learning_rate: the learning rate we want our model and optimizer to be initialized with. The use of a scheduler makes this slightly
        less impactful then it might otherwise be.
        - distributed: a bool that indicates whether or not to train in a distributed fashion if multiple gpus are detected
    Returns:
        - test_accu: the accuracy of the best performing model on the test set
        - test_loss: the loss of the best performing model on the test set
        - unique phonemes: the performance over the different phonemes on the test set
    """
    #Set up the directories for storing our results
    dir_results=Path.Path("Results")
    dir_results.mkdir(parents=True, exist_ok=True)
    #And directories for our models
    model = model_selector(model_name)
    dir_best = Path.Path("Kfold_Checkpoints")
    dir_best.mkdir(parents=True, exist_ok=True)
    dir_raw=dir_best / "{} Checkpoints".format(model.toString())
    dir_raw = Path.Path(dir_raw)
    dir_raw.mkdir(parents=True, exist_ok=True)

    #Set up for distributed training
    world_size= torch.cuda.device_count()

    #Use GPU if possible - used always for testing, sometimes for training
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    #Run distributed training if the world size is greater than 1
    if world_size > 1 and distributed==True:
        save_location=dir_raw
        save_every=4
        print("*****************")
        print("Starting Distributed training for %s model" %model_name)
        for kInt in range(5):
            mp.spawn(fn=distributed_main, args=(world_size, model_name, learning_rate, EPOCHS, save_every, save_location), nprocs=world_size, join=True, Kfold_eval=True, kInt=kInt)

            #Load in our best performing model
        filename="Kfold_Checkpoints/best_{}_checkpoint.pth.tar".format(model.toString())
        model, optimizer, Testloader=load_training_objects(model_name, learning_rate, test=True, distributed=False)
        model, optimizer=load_checkpoint(model, optimizer, torch.load(filename, weights_only=True))

        #Set up for model testing
        model.to(device)

        #Test model
        print("+---------------------------------------------+")
        print("Using Test Data. Test Accuracy :")
        test_accu, test_loss, class_correct, class_total = evaluate(model, Testloader, device=device)
        print("+---------------------------------------------+")
    
    else:
        
        print("*****************")
        print("Starting training for %s model on device %s" %(model_name, device))
        save_location=dir_raw
        save_every=4
        for kInt in range(5):
            serial_train(device, model_name, learning_rate, EPOCHS, save_every, save_location, Kfold_eval=True, kInt=kInt)
        
    print("*****************")
    print("Finished training")

def test_best(model_name: str, learning_rate: int =0.01, distributed: bool =True, Kfold_eval: bool=False, kInt: int=None, noise: bool=False, noise_type: str = "White", SNR: int=30):
    """
    This is the function that can be used to test the best model alone - and is built from the assumption that the model is already trained. 
    It loads in a model from the required save location labelled "best" and immediately tests the model. This can be changed to test the model 
    with or without added noise too.
    Args:
        - model_name: a string name identifying the model we want to train.
        - EPOCHS: the total number of epochs we want to train our model for.
        - learning_rate: the learning rate we want our model and optimizer to be initialized with. The use of a scheduler makes this slightly
        less impactful then it might otherwise be.
        - distributed: a bool that indicates whether or not to train in a distributed fashion if multiple gpus are detected
    Returns:
        - test_accu: the accuracy of the best performing model on the test set
        - test_loss: the loss of the best performing model on the test set
        - unique phonemes: the performance over the different phonemes on the test set
    """
    #Set up the directories for storing our results
    dir_results=Path.Path("Results")
    dir_results.mkdir(parents=True, exist_ok=True)
    #And directories for finding our models
    model = model_selector(model_name)
    dir_best = Path.Path("Model Checkpoints")
    dir_best.mkdir(parents=True, exist_ok=True)
    dir_raw=dir_best / "{} Checkpoints".format(model.toString())
    dir_raw = Path.Path(dir_raw)
    dir_raw.mkdir(parents=True, exist_ok=True)
    #If kfold eval is set up
    if Kfold_eval:
        dir_raw = dir_raw / "kFold Eval {}".format(kInt)
        dir_raw = Path.Path(dir_raw)
        dir_raw.mkdir(parents=True, exist_ok=True)

        dir_results=dir_results / "k-Fold Stability"
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
    #if Noise is enabled
    if noise:
        #dir_raw= dir_raw / "Noisy"
        #dir_raw = Path.Path(dir_raw)
        #dir_raw.mkdir(parents=True, exist_ok=True)
        #dir_raw= dir_raw / "{}".format(noise_type) 
        #dir_raw = Path.Path(dir_raw)
        #dir_raw.mkdir(parents=True, exist_ok=True)
        #dir_raw=dir_raw / "{}".format(SNR)
        #dir_raw = Path.Path(dir_raw)
        #dir_raw.mkdir(parents=True, exist_ok=True)

        dir_results=dir_results / "Noisy"
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
        dir_results=dir_results / "{}".format(noise_type)
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
        dir_results=dir_results / "{}".format(SNR)
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)


    #Set up for distributed training
    world_size= torch.cuda.device_count()

    #Use GPU if possible - used always for testing, sometimes for training
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        
    #Load in our best performing model
    filename=dir_raw/ "best_{}_checkpoint.pth.tar".format(model.toString())

    print("**************************************************************")
    print("Testing Best Model found at: %s" %filename)
    print("**************************************************************")
    model, optimizer, Testloader=load_training_objects(model_name, learning_rate, test=True, distributed=False, noise=noise, noise_type=noise_type, SNR=SNR)
    model, optimizer=load_checkpoint(model, optimizer, torch.load(filename, weights_only=True))

    #Set up for model testing
    model.to(device)

    #Test model. REQUIRES NOISE TESTING FUNCTIONALITY
    print("+---------------------------------------------+")
    print("Using Test Data. Test Accuracy :")
    test_accu, test_loss, class_correct, class_total, time, confusion= evaluate(model, Testloader, device=device)
    print("+---------------------------------------------+")

    #Turn the confusion matrix into a dataframe
    df_confusion = pd.DataFrame(confusion)
    #Save the confusion matrix too
    dir_confusion=dir_results / 'Confusion'
    dir_confusion.mkdir(parents=True, exist_ok=True)
    class_confusion=dir_confusion / '{}_Phoneme_Confusion_df.csv'.format(model.toString())
    df_confusion.to_csv(class_confusion, index=False)
    
    unique_phonemes=pd.read_csv("Data/Phoneme_List.csv")
    unique_phonemes['Correct']=class_correct
    unique_phonemes['Total']=class_total
                
    
    results_dict={ "test_loss": test_loss, "correct_class": class_correct, "total_class": class_total, "Runtime":time}
        
    #Save the results of the model to a JSON file
    model_result = model.toString() + ".pickle"
    results_file = dir_results/ model_result
    
    # open file for writing, "w" 
    f = open(results_file,"wb")

    # write json object to file
    pickle.dump(results_dict, f)

    # close file
    f.close()
    
    return test_accu, test_loss, unique_phonemes, time


#Combines train and evaluation to fully train up a model
def train_epochs(model_name: str, EPOCHS: int , learning_rate: int =0.01, distributed: bool =True, Kfold_eval: bool=False, kInt: int=None, noise: bool=False, noise_type: str = "White", SNR: int=30):
    """
    This is the main function in all the Train Test Functions, and calls on a selection of helper functions to acheive the required performance.
    This both trains and then immediately tests the best resulting model. Training can either be done on a single device (serial train) or 
    distributed (mp.spawn(distributed_main), based on the distributed flag and the possible detection of multiple gpus.
    Args:
        - model_name: a string name identifying the model we want to train.
        - EPOCHS: the total number of epochs we want to train our model for.
        - learning_rate: the learning rate we want our model and optimizer to be initialized with. The use of a scheduler makes this slightly
        less impactful then it might otherwise be.
        - distributed: a bool that indicates whether or not to train in a distributed fashion if multiple gpus are detected
    Returns:
        - test_accu: the accuracy of the best performing model on the test set
        - test_loss: the loss of the best performing model on the test set
        - unique phonemes: the performance over the different phonemes on the test set
    """
    #Set up the directories for storing our results
    dir_results=Path.Path("Results")
    dir_results.mkdir(parents=True, exist_ok=True)
    #And directories for our models
    model = model_selector(model_name)
    dir_best = Path.Path("Model Checkpoints")
    dir_best.mkdir(parents=True, exist_ok=True)
    dir_raw=dir_best / "{} Checkpoints".format(model.toString())
    dir_raw = Path.Path(dir_raw)
    dir_raw.mkdir(parents=True, exist_ok=True)
    #If kfold eval is set up
    if Kfold_eval:
        dir_raw = dir_raw / "kFold Eval {}".format(kInt)
        dir_raw = Path.Path(dir_raw)
        dir_raw.mkdir(parents=True, exist_ok=True)

        dir_results=dir_results / "k-Fold Stability"
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
    #if Noise is enabled
    if noise:
        dir_raw= dir_raw / "Noisy"
        dir_raw = Path.Path(dir_raw)
        dir_raw.mkdir(parents=True, exist_ok=True)
        dir_raw= dir_raw / "{}".format(noise_type) 
        dir_raw = Path.Path(dir_raw)
        dir_raw.mkdir(parents=True, exist_ok=True)
        dir_raw=dir_raw / "{}".format(SNR)
        dir_raw = Path.Path(dir_raw)
        dir_raw.mkdir(parents=True, exist_ok=True)

        dir_results=dir_results / "Noisy"
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
        dir_results=dir_results / "{}".format(noise_type)
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
        dir_results=dir_results / "{}".format(SNR)
        dir_results=Path.Path(dir_results)
        dir_results.mkdir(parents=True, exist_ok=True)
        
        
    #Set up for distributed training
    world_size= torch.cuda.device_count()

    #Use GPU if possible - used always for testing, sometimes for training
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    #Run distributed training if the world size is greater than 1
    if world_size > 1 and distributed==True:
        save_location=dir_raw
        save_every=4
        print("*****************")
        print("Starting Distributed training for %s model" %model_name)

        mp.spawn(fn=distributed_main, args=(world_size, model_name, learning_rate, EPOCHS, save_every, save_location, Kfold_eval, kInt), nprocs=world_size, join=True)

    else:
        
        print("*****************")
        print("Starting training for %s model on device %s" %(model_name, device))
        save_location=dir_raw
        save_every=4
        serial_train(device, model_name, learning_rate, EPOCHS, save_every, save_location, Kfold_eval, kInt, noise, noise_type, SNR)
        
    print("*****************")
    print("Finished training")

    #Load in our best performing model
    filename=dir_raw / "best_{}_checkpoint.pth.tar".format(model.toString())
    model, optimizer, Testloader=load_training_objects(model_name, learning_rate, test=True, distributed=False, noise=noise, noise_type=noise_type, SNR=SNR)
    model, optimizer=load_checkpoint(model, optimizer, torch.load(filename, weights_only=True))

    #Set up for model testing
    model.to(device)

    #Test model
    print("+---------------------------------------------+")
    print("Using Test Data. Test Accuracy :")
    test_accu, test_loss, class_correct, class_total, time, confusion = evaluate(model, Testloader, device=device)
    print("+---------------------------------------------+")

    #Turn the confusion matrix into a dataframe
    df_confusion = pd.DataFrame(confusion)
    #Save the confusion matrix too
    dir_confusion=dir_results / 'Confusion'
    dir_confusion.mkdir(parents=True, exist_ok=True)
    class_confusion=dir_confusion / '{}_Phoneme_Confusion_df.csv'.format(model.toString())
    df_confusion.to_csv(class_confusion, index=False)
    
    unique_phonemes=pd.read_csv("Data/Phoneme_List.csv")
    unique_phonemes['Correct']=class_correct
    unique_phonemes['Total']=class_total
                
    
    results_dict={ "test_loss": test_loss, "correct_class": class_correct, "total_class": class_total, "Runtime":time}
        
    #Save the results of the model to a JSON file
    model_result = model.toString() + ".pickle"
    results_file = dir_results/ model_result
    
    # open file for writing, "w" 
    f = open(results_file,"wb")

    # write json object to file
    pickle.dump(results_dict, f)

    # close file
    f.close()
    
    return test_accu, test_loss, unique_phonemes, time