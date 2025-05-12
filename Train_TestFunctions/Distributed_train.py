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

This holds functions and classes used for training our models in a distributed fashion.
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

from Encoders import FeatureEncoders
from .train_init import model_selector, load_train_settings, load_training_objects, prepare_dataloader, load_checkpoint

def find_free_port():
    """ This finds a free port on the device, if needed. Code from 
    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    Returns:
     - port_name : str
    """
    
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def ddp_setup(rank, world_size):
    """
    This sets up the workers and co-ordinates communication across processes.
    This should allow training over multiple GPUs
    Args: 
        - rank: Unique identifier of each process
        - world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="50197"

    print(">> Setting up system for distributed training")
    print(f"Initialising GPU {rank+1} of {world_size}")
    #The backend is set up for nvidia gpus (cuda)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    
class Trainer:
    """
    This Trainer class was developed to train models in a distributed fashion. Inspired and adapted from 
    code shown in https://www.youtube.com/watch?v=-LAtx9Q6DA8&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=3 
    Designed to be used in conjunction with distributed main.
    Args: 
        - model: the torch Neural Network model to be trained
        - train_data : the training data dataloader
        - optimizer: the optimiser to be used for assisting in training
        - gpu_id : the device that the train instance is being run on
        - save_every : the epoch interval where the model is saved and tested on the validation set
        - save_location : location where the model checkpoints and training data should be saved to
        - scheduler:  the learning rate scheduler to be used in training
        - num_classes: the number of classes that exist in the output
        - validate_data: the validation dataset dataloader
    """
    def __init__(self,
                model: torch.nn.Module,
                train_data: DataLoader,
                optimizer: torch.optim.Optimizer,
                gpu_id: int,
                save_every: int, 
                save_location: str, 
                scheduler= torch.optim.lr_scheduler,
                num_classes: int = 39,
                validate_data: DataLoader= None,
                ) -> None:
                    self.gpu_id = gpu_id
                    self.model = model.to(gpu_id)
                    self.train_data = train_data
                    self.valid_data= validate_data
                    self.num_classes= num_classes
                    self.optimizer = optimizer
                    self.save_every = save_every
                    self.save_location = save_location
                    self.scheduler=scheduler
                    self.model = DDP(self.model, device_ids=[self.gpu_id])
                    self.best_accu=None
                    self.total_accu=None
                    #This is used to show how much better we are getting at learning individual classes
                    self.phoneme_rate=pd.read_csv("./Data/Phoneme_List.csv")
                    
                    
    def _run_batch(self, source, targets):    
        """
        Runs a training batch, calculating and updating the weights using backprop.
        Args:
            - source: the data to be input into the model
            - targets: the correct data labels to be compared against
        """
        #Pass the source data through the model and calculate backprop
        output = self.model(source)

        #Set up the optimizer to have zero grad before computing backprop
        self.optimizer.zero_grad()
        #Targets are packed sequences - unpack them and permute to be on 1st dimension
        phonemes, lens = torch.nn.utils.rnn.pad_packed_sequence(targets,
                                                               batch_first=True)
        targets=phonemes.permute(0, 2, 1)

        #Calculate loss and backprop
        loss= torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        
        #Update the weights with backprop
        self.optimizer.step()

    def _run_test(self, source, targets, correct, total, val_loss, class_correct, class_total):
        """
        Runs a validation test batch.
        Args:
            - source: the data to be input into the model
            - targets: the correct data labels to be compared against
            - correct: number of correctly labelled outputs
            - total: total number of labelled outputs
            - val_loss: validation loss over the epoch
            - class_correct: number of correctly labelled outputs per clas
            - clas_total: number of total labelled outputs per class
        Returns: 
            - correct : torch.tensor(int)
            - total: torch.tensor(int)
            - val_loss: torch.tensor(float)
            - class_correct: torch.tensor(len(num_classes), float)
            - class_total: torch.tensor(len(num_classes), float)
        """
        with torch.no_grad():
            #Pass the source data through the model and calculate backprop
            output = self.model(source)

            #Targets are packed sequences - unpack them and permute to be on 1st dimension
            phonemes, lens = torch.nn.utils.rnn.pad_packed_sequence(targets,
                                                               batch_first=True)
            targets=phonemes.permute(0, 2, 1)

            #Calculate loss and backprop
            loss= torch.nn.CrossEntropyLoss()(output, targets)
            val_loss+=loss.float()

        #Get the longest length of phonemes
        target=torch.argmax(targets.data, 1)
        predicted = torch.argmax(output.data, 1)
            
        #Check if the labels were correct for each signal in the batch
        for ii in range(phonemes.size(0)):
         #   #For every phoneme in the utterance
            for jj in range(phonemes.size(2)):
                #Check that the value isn't a pad value
                if int(jj) < int(lens[ii]) :
                    #Update total
                    total +=1
                    class_total[target[ii][jj]] += 1
                    if target[ii][jj] == predicted[ii][jj]:
                        correct +=1
                        class_correct[target[ii][jj]] += 1

        #Return accruracy, validation loss, the number of correctly defined classes and the total number 
        return correct, total, val_loss, class_correct, class_total

    def _test_epoch(self, epoch: int):
        """
        This runs over the validation set to indicate how well we are doing.
        Args:
            - Epoch : the current epoch of training
        """
        b_sz = len(next(iter(self.valid_data))[0])
        print(f"Testing For: [GPU{self.gpu_id+1 }] Epoch {epoch +1} | Batchsize {b_sz} | Steps: {len(self.valid_data)}", file=sys.stdout)

        #Set the model into evaluation mode
        self.model.module.eval()

        #Store the number of corrects, totals and validation loss over the epoch
        correct=torch.tensor(0, device=self.gpu_id).requires_grad_(False)
        total=torch.tensor(0, device=self.gpu_id).requires_grad_(False)
        val_loss=torch.tensor(0.0, device=self.gpu_id).requires_grad_(False)
        class_correct= torch.zeros(self.num_classes, device=self.gpu_id).requires_grad_(False)
        class_total=torch.zeros(self.num_classes, device=self.gpu_id).requires_grad_(False)

        for source, targets in self.valid_data:
            #Pass the sources and targets to device and run test
            source = source.to(self.gpu_id)
            targets= targets.to(self.gpu_id)
            correct, total, val_loss, class_correct, class_total= \
            self._run_test(source, targets, correct, total, val_loss, class_correct, class_total)

        dist.all_reduce(val_loss, dist.ReduceOp.SUM, async_op=True)
        dist.all_reduce(correct, dist.ReduceOp.SUM, async_op=True)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=True)
        dist.all_reduce(class_correct, dist.ReduceOp.SUM, async_op=True)
        dist.all_reduce(class_total, dist.ReduceOp.SUM, async_op=True)
        
        print(f"Correct value. Should match {correct} on GPU {self.gpu_id}")
        #Calculate the overall accuracy for the epoch and print
        accuracy=correct.cpu()/total.cpu()
        print(f"Validation accuracy for GPU {self.gpu_id+1 } Epoch {epoch + 1}: {accuracy: 0.4f}, Loss : {val_loss: 0.4f}" )

        #Keep track of how much better we are getting
        if self.gpu_id==0:
            #How much better are we getting at classes?
            column_name="Epoch {} Class Accuracies".format(epoch+1)
            #Division done in this way to avoid divide by zero error
            class_correct=class_correct.cpu()
            class_total=class_total.cpu()
            class_accu = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total!=0)
            self.phoneme_rate[column_name]=class_accu

            #Save best model, or new learning rate value if the model got worse
            if self.total_accu is not None and self.total_accu > accuracy:
                self.scheduler.step()
            else:
                self.total_accu = accuracy
            if ((self.best_accu == None) or (accuracy > self.best_accu)):
                self._save_checkpoint(epoch, best=True)
        
    def _run_epoch(self, epoch: int):
        """
        Runs an epoch of model training.
        Args:
            - Epoch : current epoch of training
        """
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id + 1}] Epoch {epoch + 1} | Batchsize {b_sz} | Steps: {len(self.train_data)}", file=sys.stdout)

        self.model.module.train()
            
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
              
    def train(self, max_epochs: int):
        """
        Trains the model for a number of epochs, testing and saving the model at given intervals.
        Args: 
            - self: instance oif trainer class
            - max_epochs: the maximum number of epochs you want to train the model for
        """
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            
            if epoch%self.save_every == 0:
                #Test the model on validation data
                if self.valid_data is not None:
                    #continue
                    self._test_epoch(epoch)
                #Save (no duplicates)
                if self.gpu_id==0:
                    self._save_checkpoint(epoch)

        #Save how class accuracies changed over time for inspection
        if self.gpu_id==0:
            class_accuracies=self.save_location / 'Class_accuracy_df.csv'   
            self.phoneme_rate.to_csv(class_accuracies, index=False)
    
                
    def _save_checkpoint(self, epoch: int, best : bool =False):
        """
        Saves a snapshot of the model and optimizer states at a certain epoch, or if it is the best performing model so far.
        Args:
            - self: Trainer class instance
            - epoch: current epoch of training
            - best: bool indicating whether the model is the best performing so far
        """
        checkpoint={"state_dict": self.model.module.state_dict() , "optimizer": self.optimizer.state_dict()}
        if best:
            location=self.save_location / "best_{}_checkpoint.pth.tar".format(self.model.module.toString())
            torch.save(checkpoint, location)
            print(f"Best Model Updated at Epoch {epoch} | Training checkpoint saved at {location}", file=sys.stdout)
        
        location=self.save_location / "Epoch_{}_{}_checkpoint.pth.tar".format(epoch, self.model.module.toString())
        torch.save(checkpoint, location)
        print(f"Epoch {epoch} | Training checkpoint saved at {location}", file=sys.stdout)


def distributed_main(rank: int,
                     world_size: int, 
                     model_name: str, 
                     learning_rate: int,
                     total_epochs: int,
                     save_every : int,
                     save_location: str,
                    Kfold_eval : bool =False,
                    kInt: int = None):
    """
    When finished: should have a trained neural network model.
    This  is the main function - from here you can run set up, load the required training objects, 
    train the model and destroy the workers when finished by a single call.
    The arguments are minimal - just strings and ints - to reduce overhead at initilisation, and allow for 
    better modulisation.
    Args:
        - rank: Unique identifier of each distributed process
        - world_size: Total of processes
        - model_name: The name of the model, so that the model itself can be loaded in
        - learning_rate: the learning rate of the model initially
        - total_epochs: the number of epochs the model should be trained for
        - save_every: the interval at whihc a model is saved and evaluated on the validation dataset
        - save_location: where the model checkpoints and class accuracies dataframe should be stored
        - Kfold_eval: whether or not this is a kFold stability test
        - KInt: If this is a kFold test, what kFold set is the validation for this run
    """
    
    #Set up the distributed system
    ddp_setup(rank, world_size)

    model, optimizer, scheduler, trainloader, valid_loader = load_training_objects(model_name, learning_rate, Kfold_eval=Kfold_eval, kInt=kInt)
    
    #Create the trainer and train
    trainer= Trainer(model, trainloader, optimizer, rank, save_every, save_location, scheduler, validate_data=valid_loader)

    trainer.train(total_epochs)
    destroy_process_group()