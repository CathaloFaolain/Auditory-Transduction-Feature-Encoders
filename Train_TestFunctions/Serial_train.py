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

This holds functions and classes used for training our models onm a single device, either cpu or gpu.
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

from .train_init import model_selector, load_train_settings, load_training_objects, prepare_dataloader, EarlyStopper, load_checkpoint


def train(model, optimizer,  train_data,  epoch,  clip_thresh=1.0, device='cuda:0'):
    """
    Runs a training epoch, loading in the data, calculating and updating the weights using backprop
    Args:
        - model: the model to be trained
        - optimizer: the optimizer to help with backprop
        - train_data: the training dataset dataloader
        - epoch: the current training epoch
        - clip_thresh: the threshold at which to clip gradients
        - device: the device, either cpu or gpu, that the model is running on
    Returns:
        - train_losses : list of floats
    """
    b_sz = len(next(iter(train_data))[0])
    print(f"[Device {device}] Epoch {epoch + 1} | Batchsize {b_sz} | Steps: {len(train_data)}")
    #Put the model in training mode
    model.train()

    #We will provide an indication of where we are in the epoch owing to likely slow speed
    indicator=int(len(train_data)/10)

    train_losses=[]
    #Go through the dataloader, getting the signals and the phonemes
    for i, data in enumerate(train_data):
        
        signals, phonemes = data

        signals=signals.to(device)
        phonemes=phonemes.to(device)
        
        #Predict the label for the given audio signals
        predicted_phoneme = model(signals)
        
        optimizer.zero_grad()
        
        phonemes, lens = torch.nn.utils.rnn.pad_packed_sequence(phonemes,
                                                               batch_first=True)
        phonemes=phonemes.permute(0, 2, 1)
        
        loss = torch.nn.CrossEntropyLoss()(predicted_phoneme, phonemes)
        train_losses.append(loss.item())
        
        #Essential backprop
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
        #Perform the optimisation step
        optimizer.step()
        
        #Let us know where in the epoch we are. Print out . for every 10% 
        if i%(indicator) == 0:
            print(".",end="")
            sys.stdout.flush()
       
    return train_losses

#This is used to evaluate the models
def evaluate(model,  valid_data, num_classes=39, device='cuda:0'):
    """
    This function is used for evaluating the dataset over a validation or test set. This has slightly more detail than 
    its distributed cousin - as it also records the model runtime.
    Args:
        - model : the model being evaluated
        - valid_data : the evaluating dataset's dataloader
        - num_classes: the numbner of classes in the output. Defaults to 39
        - device : the device on which the model is run. Defaults to cuda (gpu)
    """
    b_sz = len(next(iter(valid_data))[0])
    print(f"Testing For: | Batchsize {b_sz} | Steps: {len(valid_data)}")
    total_time=0
    
    correct = 0
    total = 0
    val_losses=[]
    
    #Set the model into evaluation mode
    model.eval()
    
    #Keep a running total of how good we are at getting each class
    corrects= np.zeros(num_classes)
    totals=np.zeros(num_classes)
    confusion=np.zeros((num_classes, num_classes))

    
    start_time=time.time()
    with torch.no_grad():
        for data in valid_data:
            signals, phonemes = data

            
            signals=signals.to(device)
            phonemes=phonemes.to(device)

            
            phonemes, lens = torch.nn.utils.rnn.pad_packed_sequence(phonemes, batch_first=True)
            
            outputs = model(signals)
        
            phonemes=phonemes.permute(0, 2, 1)
            #Get the class labels
            loss = torch.nn.CrossEntropyLoss()(outputs, phonemes)
            val_losses.append(loss.item())
            
            #Get the longest length of phonemes
            target=torch.argmax(phonemes.data, 1)
            predicted = torch.argmax(outputs.data, 1)
            
            #Check if the labels were correct for each signal in the batch
            for ii in range(phonemes.size(0)):
                #For every phoneme in the utterance
                for jj in range(phonemes.size(2)):
                    #Check that the value isn't a pad value
                    if int(jj) < int(lens[ii]) :
                        #Update total
                        total +=1
                        totals[target[ii][jj]] += 1
                        if target[ii][jj] == predicted[ii][jj]:
                            correct +=1
    
                            corrects[target[ii][jj]] += 1
                        else:
                            confusion[target[ii][jj]][predicted[ii][jj]] += 1
                            
    end_time=time.time()
    total_time=end_time-start_time
    
    val_losses=np.mean(val_losses)
    print(f"Evaluation accuracy: {(correct/total): 0.4f}, Phoneme Error Rate: {(np.sum(total-correct)/np.sum(total)): 0.4f}, Loss : {val_losses: 0.4f}, Time: {end_time-start_time : 0.4f}s, Time per sample: {(total_time)/len(valid_data): 0.4f}s" )

    return (correct/total), val_losses, corrects, totals, total_time, confusion

def serial_train(device: str, 
                 model_name: str, 
                 learning_rate : int,
                 total_epochs: int,
                 save_every : int,
                 save_location: str,
                 Kfold_eval: bool, 
                 kInt: int, 
                noise: bool,
                noise_type: str,
                SNR: int):
    
    """
    When finished: should have a trained neural network model.
    This  is the main function - from here you can run set up, load the required training objects, 
    train the model and test validation when finished by a single call.
    The arguments are minimal - just strings and ints - to reduce overhead at initilisation, and allow for 
    better modulisation.
    Args:
        - device: the device (either gpu or cpu) to run the model on
        - model_name: The name of the model, so that the model itself can be loaded in
        - learning_rate: the learning rate of the model initially
        - total_epochs: the number of epochs the model should be trained for
        - save_every: the interval at whihc a model is saved and evaluated on the validation dataset
        - save_location: where the model checkpoints and class accuracies dataframe should be stored
        - Kfold_eval: whether or not this is a kFold stability test
        - KInt: If this is a kFold test, what kFold set is the validation for this run
    Returns: 
        - Epoch_train_loss: list of floats of average training loss per epoch
        - Epoch_val_loss: list of floats of average validation loss per epoch
    """
    print(">> Setting: Training in Serial")
    #The early stopper
    early_stopper=EarlyStopper(patience=5)
    total_accu = None
    best_accu = None
    
    #Get our model, optimizer, scheduler, trainloader and validloader set up and ready
    model, optimizer, scheduler, trainloader, valid_loader=load_training_objects(model_name, learning_rate, distributed=False, Kfold_eval=Kfold_eval, kInt=kInt, noise=noise, noise_type=noise_type, SNR=SNR)
    model=model.to(device)

    #We will use this to keep track of how much we are improving per class over the epochs
    phoneme_rate=pd.read_csv("Data/Phoneme_List.csv")
    epoch_train_loss, train_loss, epoch_val_loss, epoch_PER, epoch_accu, val_time, save_epoch= [],[], [],[], [], [],[]
        
    #Go through all the epochs and train the models
    for epoch in range(total_epochs):
        #This will hold the training and validation losses by epoch
        train_losses=train(model, optimizer, trainloader,  epoch, device=device)
        epoch_train_loss.append(np.mean(train_losses))
        
        #We will test the model on the valdation set every few epochs
        if epoch%save_every == 0:
            accu_val, val_losses, class_correct, class_total, total_time, confusion= evaluate(model, valid_loader, device=device)
            
            #We will save our train losses and val losses at this time too, as well as the val PER
            train_loss.append(np.mean(train_losses))
            epoch_val_loss.append(val_losses)
            epoch_PER.append(np.sum(class_total-class_correct)/np.sum(class_total))
            epoch_accu.append(accu_val)
            val_time.append(total_time)
            save_epoch.append(epoch)
            
        
            #Do the scheduler step if the model got worse
            if total_accu is not None and total_accu > accu_val:
                scheduler.step()
            else:
                total_accu = accu_val
            #We will save the model with the best validation accuracy for later loading
            if best_accu is None or best_accu < accu_val:
                #Save the model 
                print("Saving best model")
                checkpoint={"state_dict": model.state_dict() , "optimizer": optimizer.state_dict()}
                filename= save_location / "best_{}_checkpoint.pth.tar".format(model.toString())
                torch.save(checkpoint, filename)
                #Update the best accuracy
                best_accu = accu_val
                
           
            #Check if early stopping is required
            if early_stopper.early_stop(val_losses):
                print("Early stopping due to validation loss increasing")
                break
                
            #Save the model at the checkpoint
            checkpoint={"state_dict": model.state_dict() , "optimizer": optimizer.state_dict()}
            filename= save_location / "Epoch_{}_{}_checkpoint.pth.tar".format(epoch+1, model.toString())
            torch.save(checkpoint, filename)
            
            #How much better are we getting at classes?
            column_name="Epoch {} Phoneme Class Error Rate".format(epoch+1)
            phoneme_rate[column_name]=(class_total-class_correct)/class_total
            #What are they getting confused with?
            
            #column_name="Epoch {} Class Total".format(epoch+1)
            #phoneme_rate[column_name]=class_total
            
    
    class_accuracies=save_location / 'Class_accuracy_df.csv'  
    phoneme_rate.to_csv(class_accuracies, index=False)
    #class_confusion=save_location / 'Phoneme_Confusion_df.csv'
    #confusion_matrix.to_csv(class_confusion, index=False)
    training_location=save_location / 'TrainingLoss_PER_df.csv'
    training_df=pd.DataFrame({"Epoch Value": save_epoch,"Epoch PER": epoch_PER, "Epoch accuracy" :epoch_accu,"Epoch Val Loss": epoch_val_loss, "Epoch Train Loss":train_loss, "Test Time": val_time})
    training_df.to_csv(training_location, index=False)
    

    return epoch_train_loss, epoch_val_loss