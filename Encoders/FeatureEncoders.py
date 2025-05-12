#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 12:08 2024

@author: Cathal Ó Faoláin

The goal of this work is to understand how we can use predicted IHC potentials,
such as those predicted by WavIHC, introduced in the paper 
"WaveNet-based approximation of a cochlear filtering and hair cell transduction model". 
Feature encoders designed to use these predicted IHC potentials are evaluated against 
other state-of-the-art feature encoders in order to understand how discriminating they are,
and over a range of different Signal-to-Noise Ratios (SNRs).

This holds the feature encoders that we shall evaluate, along with 
any helper functions they require. We have 5 feature encoders:

- Contrastive Predictive Coding (CPC) 
- Wav2vec2.0
- Autoregressive Predictive Coding (APC)
- IHC CPC
- IHC Wav2vec2 
- Whisper (OpenAI)

The first three feature encoders, CPC, Wav2vec2.0 and APC are based on the designs used in each of the papers.
Any context encoders that tries to model longer-term dependencies have been removed - so no transformers or
Recurrent Neural Networks (RNN). This is to allow for us to evaluate how discriminating the features themselves 
are. 
IHC CPC and Wav2vec2 are adapted feature encoders that take predicted IHC potentials as input rather than the signal
alone. Each is inspired by their namesake models.
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
from collections import namedtuple

#IMPORT THE IHC MODEL (WAVIHC)
sys.path.append('./IHCApproxNH/')
from classes import WaveNet
from utils import utils


#######################################################################################################
#APC MODELS AND HELPER FUNCTIONS
#######################################################################################################

class MLP(nn.Module):
     
    """MLP is a multi-layer fully-connected network with ReLU activations.
    During training and testing (i.e., feature extraction), each input frame is
    passed into MLP. This shallow neural network is used here to access some non-linear feature information
    that should help with phoneme discrimination.
  
    Batch normalisation is used in order to improve the stability of the learning. 
    """
    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super(MLP, self).__init__()
        #Get the input and output sizes set up before initialising layers
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        #Get the layers initalised as a list of input, output sizes
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=in_size, out_features=out_size)
            for (in_size, out_size) in zip(input_sizes, output_sizes)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, mel_dim)
        for layer in self.layers:
            inputs = self.dropout(self.relu(layer(inputs)))
            
        return inputs
    
class MelSimple(nn.Module):
    """MelSimple is a neural network designed for phoneme prediction using Mel Spectrograms as input
    The goal is to understand how discriminative mel spectrograms are as one of the most commonly used speech
    features used in cutting edge models. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is analogous to the feature encoder stage of APC.
    
    MelSimple consists of an optional MLP stage. The MLP stage can consist of either 1 or 3 
    """
    
    def __init__(self, mel_dim, layers: bool = False, n_classes=39):
        super(MelSimple, self).__init__()
        self.mel_dim = mel_dim
            

        #Initialise the MLP if it exists, and make sure the input size matches the 
        #Mel size 
        if layers:
            PrenetConfig = namedtuple(
              'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])
        
            mlp_config=PrenetConfig(
              80, 512, 3, 0.0)
          # Make sure the dimensionalities are correct
            assert mlp_config.input_size == mel_dim
            hidden_dim=mlp_config.hidden_size
            self.mlp = MLP(
                input_size=mlp_config.input_size,
                num_layers=mlp_config.num_layers,
                hidden_size=mlp_config.hidden_size,
                dropout=mlp_config.dropout)
        else:
            self.mlp = None
            hidden_dim=mel_dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(hidden_dim, n_classes)

    def toString(self):
        if self.mlp is None:
            return "MelSimple"
        else:
            return "MelSimple_MLP"
        
    
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        #Unpack the sequence
        x, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        
        #print(x.shape)
    
        #If the MLP is configured, run it
        if self.mlp is not None:
            outputs = self.mlp(x)
            # outputs: (batch_size, seq_len, output_size)
        else:
            outputs = x

        #Predict the phoneme using mel
        predicted_mel = self.fc(outputs)
        
        predicted_phone=predicted_mel.permute(0, 2, 1)

        return predicted_phone

###################################################################################################################################################
# Whisper MODEL AND HELPER FUNCTIONS
###################################################################################################################################################


        
class WhisperEncoder(nn.Module):
    """This is the Encoder of OpenAI Whisper, and has been adapted to omit the context encoding layers. 
    This is based on the feature encoder described in the paper: "Robust Speech Recognition via Large-Scale
    Weak Supervision", and the code is taken from: https://github.com/openai/whisper/blob/main/whisper/model.py
    The goal is to understand how discriminative the output features of the encoder are, without any context being included.
    This works on the assumption that linear separability defines the accessibility of information for 
    downstream tasks. This feature encoder uses convolutions, and thus can model some local dependencies
    and context, on top of Mel Spectrograms.

    Values are set to defaults, 80 mel spectrogram and n_state of 512. N_ctx is removed as our system allows for dynamic padding,
    removing the requirement for setting a constant sequence length size.
    """
    def __init__(
        self, n_mels: int =80 , n_state: int=512,  n_classes=39,
    ):
        super().__init__()
        self.n_state=n_state
        self.n_mels=n_mels
        self.n_classes=n_classes
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        #We are retraining from scratch - therefore we require initialised weights, and will use
        #kaiming following current best practice
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        
        #Simple classification layer added
         #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(n_state, self.n_classes)
        
    def toString(self):
        if self.n_state==512:
            return "Whisper"
        else:
            return "Whisper_{}".format(self.n_state)
            
    def forward(self, inputs):
        """
        Input:
        inputs: (Batch_size, seq_len, n_mels)

        """
        #Our training and testing set up is slightly different, and requires different loading
        #model expects:
        #x : torch.Tensor, shape = (batch_size, n_mels, seq_len)
        #    the mel spectrogram of the audio
        #Unpack the sequence        
        x, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        #Achieved via permutation
        x=x.permute(0, 2, 1)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        #Requires other format to predict classes
        x = x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y
        

    
####################################################################################################################################################
# CPC MODELS AND HELPER FUNCTIONS
####################################################################################################################################################
class CpcEncoder(nn.Module):
    """CpcEncoder is a neural network designed for phoneme prediction using the signal alone as input.
    As the name implies, this is based on the feature encoder described in the paper:
    "Representation Learning with Contrastive Predictive Coding". The goal is to
    understand how discriminative the output features of the encoder are, without any context being included.
    This works on the assumption that linear separability defines the accessibility of information for 
    downstream tasks. This feature encoder uses convolutions, and thus can model some local dependencies
    and context.
    
    CpcEncoder has a fully connected classification stage built on top of it.

    """
    def __init__(self, nHiddenUnits=512, nLayers=5, dropout: float = 0.0, conv_bias: bool = False, n_classes=39):
        super().__init__()
        
        self.nHiddenChannels=nHiddenUnits
        self.strides=[5, 4, 2, 2, 2]
        self.kernels=[10, 8, 4, 4, 4]
        self.nLayers=nLayers
        self.n_classes=n_classes
        
        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                #The padding for conv1d cpc is padding=stride/2, given that k is always 2*stride
                p=math.ceil(stride/2)
                #Make the convolutional layer, and initialise it well
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=p, padding_mode='zeros', bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            #Return the convolutional, dropout and activation function layer 
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.ReLU())
            
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        
        in_d=1
        for iLayer in range(self.nLayers):
            #Append the temporal convolution block
            self.conv_layers.append(
                block(
                    in_d,
                    self.nHiddenChannels,
                    self.kernels[iLayer],
                    self.strides[iLayer],
                    conv_bias=conv_bias, ) )
                                 
            #Set the in dimension to be the number of hidden units, i.e. 512
            #After the first layer
            in_d=self.nHiddenChannels
            
         #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(self.nHiddenChannels, self.n_classes)
     
    def toString(self):
        if self.nHiddenChannels==512:
            return "CPC_Encoder"
        else:
            return "CPC_Encoder_{}".format(self.nHiddenChannels)
    
    def forward(self, x):
        """Forward function for both training and testing.

        input:
          x: pack_padded_sequence(batch_size, seq_len, n_channels)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        
        #Unpack my little sardines and change them to have the channel dimension in the middle
        x, self.lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x=x.permute(0, 2, 1)
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y

##################################################################################################################################################
#WAV2VEC2.0 MODEL AND HELPER FUNCTIONS
###################################################################################################################################################
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    
class Wav2vec2Encoder(nn.Module):
    """Wav2vec2Encoder is a neural network designed for phoneme prediction using the signal alone as input.
    As the name implies, this is based on the feature encoder described in the paper:
    "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations". The goal is to
    understand how discriminative the output features of the encoder are, without any context being included.
    This works on the assumption that linear separability defines the accessibility of information for 
    downstream tasks. This feature encoder uses convolutions, and thus can model some local dependencies
    and context.
    
    Wav2vec2Encoder has a fully connected classification stage built on top of it.
    
    This code is taken from "https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L844"
    Comments added by Cathal Ó Faoláin. Original class is called 'class ConvFeatureExtractionModel(nn.Module)' in code.
    Initialisation slightly changed to make changing the number of hidden units easier.
    """
    def __init__(self, nHiddenUnits=512, dropout: float = 0.0, mode: str = "default",
        conv_bias: bool = False,  required_seq_len_multiple=2, n_classes=39):
        super().__init__()

        #This is the wav2vec2 style feature encoder
        self.nHiddenChannels=nHiddenUnits
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        self.n_classes=n_classes
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 1
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

        #The final fully connected softmax layer for seq2seq prediction
        self.fc =  nn.Linear(self.nHiddenChannels, self.n_classes)
            
    def toString(self):
        if self.nHiddenChannels==512:
            return "Wav2Vec2.0_Encoder"
        else:
            return "Wav2Vec2.0_Encoder_{}".format(self.nHiddenChannels)
    
    def forward(self, x):
        """Forward function for both training and testing.

        input:
          x: pack_padded_sequence(batch_size, seq_len, n_channels)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        # BxT -> BxCxT
        #x = x.unsqueeze(1)
        #Unpack my little sardines and change them to have the channel dimension in the middle
        x, self.lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x=x.permute(0, 2, 1)
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

        #Change the last dimension to be the channel
        #Lets test without final permute back
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        #print(y.shape)
        
        return y
    
###########################################################################################################################################
#******************************************************IHC-BASED MODELS*******************************************************************#
###########################################################################################################################################

###########################################################################################################################################
#IHC CPC AND HELPER FUNCTIONS
###########################################################################################################################################
class IHC_Cpc(nn.Module):
    """IHC CPC is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is analogous to the feature encoder stage of CPC, altered to use IHC potentials rather the signal alone.
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=512, nLayers=5, dropout: float = 0.0, conv_bias: bool = False):
        super().__init__()
        #######################
        #Configure the WavIHC model
        # load configuration file  
        with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml",'r') as ymlfile:
            conf = yaml.safe_load(ymlfile)   #
            self.conf=conf
            
        #Constants
        self.sigMax = torch.tensor(55)
        self.ihcogramMax = torch.tensor(1.33)
        self.ihcogramMax = utils.comp(self.ihcogramMax, conf['scaleWeight'], conf['scaleType'])
        self.fs=16000
        
        # number of samples to be skipped due to WaveNet processing    
        self.skipLength = (2**conf['nLayers'])*conf['nStacks']      
    
        ## initialize WaveNet and load model paramaters
        self.NET = WaveNet.WaveNet(conf['nLayers'],
                                   conf['nStacks'],
                                   conf['nChannels'],
                                   conf['nResChannels'],
                                   conf['nSkipChannels'],
                                   conf['numOutputLayers'])
        
        self.NET.load_state_dict(torch.load("./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",\
                                        map_location=torch.device('cuda:0'), weights_only=True))
        
        # Freeze the IHC layers
        for param in self.NET.parameters():
            param.requires_grad= False
        #self.frame_shift = int(segdur*self.fs)
        #self.frame_len = self.frame_shift + self.skipLength
        ############################
            
        #This is the CPC style feature encoder
        self.nHiddenChannels=nHiddenUnits
        self.strides=[5, 4, 2, 2, 2]
        self.kernels=[10, 8, 4, 4, 4]
        self.nLayers=nLayers
        self.n_classes=n_classes
        
        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                #The padding for conv1d cpc is padding=stride/2, given that k is always 2*stride
                p=math.ceil(stride/2)
                #Make the convolutional layer, and initialise it well
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=p, padding_mode='zeros', bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            #Return the convolutional, dropout and activation function layer 
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.ReLU())
            
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        
        in_d=80
        for iLayer in range(self.nLayers):
            #Append the temporal convolution block
            self.conv_layers.append(
                block(
                    in_d,
                    self.nHiddenChannels,
                    self.kernels[iLayer],
                    self.strides[iLayer],
                    conv_bias=conv_bias, ) )
                                 
            #Set the in dimension to be the number of hidden units, i.e. 512
            #After the first layer
            in_d=self.nHiddenChannels
            
         #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(self.nHiddenChannels, self.n_classes)
        
    def toString(self):
        if self.nHiddenChannels==512:
            return "IHC_CPC"
        else:
            return "IHC_CPC_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        ##################################################################################
        # WAVENET AUDITORY TRANSDUCTION RUNNER
        ##################################################################################
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        signals=signals.permute(0, 2, 1)
        
        with torch.no_grad():
            IHC_predicted=self.NET(signals)
        
            IHC_predicted = IHC_predicted*self.ihcogramMax
            x = utils.invcomp(IHC_predicted, self.conf['scaleWeight'], self.conf['scaleType'])
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)#

        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y


    
######################################################################################################################################################
# WAV2VEC2.0 MODEL AND HELPER FUNCTIONS
######################################################################################################################################################
class IHC_Wav2vec2(nn.Module):
    """IHC_Wav2vec2 is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is analogous to the feature encoder stage of Wav2vec2.0, altered to use IHC potentials rather than the 
    signal alone.
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=512, dropout: float = 0.0, mode: str = "default", \
                 required_seq_len_multiple=2, conv_bias: bool = False):
        super().__init__()
        self.n_classes=n_classes
        #######################
        #Configure the WavIHC model
        # load configuration file  
        with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml",'r') as ymlfile:
            conf = yaml.safe_load(ymlfile)   #
            self.conf=conf
            
        #Constants
        self.sigMax = torch.tensor(55)
        self.ihcogramMax = torch.tensor(1.33)
        self.ihcogramMax = utils.comp(self.ihcogramMax, conf['scaleWeight'], conf['scaleType'])
        self.fs=16000
        
        # number of samples to be skipped due to WaveNet processing    
        self.skipLength = (2**conf['nLayers'])*conf['nStacks']      
    
        ## initialize WaveNet and load model paramaters
        self.NET = WaveNet.WaveNet(conf['nLayers'],
                                   conf['nStacks'],
                                   conf['nChannels'],
                                   conf['nResChannels'],
                                   conf['nSkipChannels'],
                                   conf['numOutputLayers'])
        
        self.NET.load_state_dict(torch.load("./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",\
                                        map_location=torch.device('cuda:0'), weights_only=True))
        
        # Freeze the IHC layers
        for param in self.NET.parameters():
            param.requires_grad= False
        #self.frame_shift = int(segdur*self.fs)
        #self.frame_len = self.frame_shift + self.skipLength
        ############################
        self.nHiddenChannels=nHiddenUnits
        #This is the wav2vec2 style feature encoder
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 80
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(nHiddenUnits, self.n_classes)
       
    def toString(self):
        if self.nHiddenChannels==512:
            return "IHC_Wav2Vec2"
        else:
            return "IHC_Wav2Vec2_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        ##################################################################################
        # WAVENET AUDITORY TRANSDUCTION RUNNER
        ##################################################################################
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        signals=signals.permute(0, 2, 1)
        
        with torch.no_grad():
            IHC_predicted=self.NET(signals)
        
            IHC_predicted = IHC_predicted*self.ihcogramMax
            x = utils.invcomp(IHC_predicted, self.conf['scaleWeight'], self.conf['scaleType'])
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y
        
##########################################################################################################
# RESNET-IHC-CPC AND HELPER FUNCTIONS
##########################################################################################################
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Implentation here taken from https://github.com/osmr/imgclsmob/blob/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models/common.py#L731,
    and adapted to 1d convolutions

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 ):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=mid_cannels,
            kernel_size=1, bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=mid_cannels,out_channels=channels,
            kernel_size=1, bias=True)
        self.sigmoid =  nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x
        

#IHC EXTRACT
class IHC_Extract(nn.Module):
    """IHC extract is a neural network designed for phoneme prediction using signal as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is designed to use the IHC potentials. Based on an adapted version of CPC due to the lower inital parameters
    and comparable performance, this model has a number of improvements that should hopefully improve the model 
    performance overall. 
    
    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training.

    Next, a squeeze and excitation layer was added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, nLayers=5, conv_bias: bool = False):
        super().__init__()
         #######################
        #Configure the WavIHC model
        # load configuration file  
        with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml",'r') as ymlfile:
            conf = yaml.safe_load(ymlfile)   #
            self.conf=conf
            
        #Constants
        self.sigMax = torch.tensor(55)
        self.ihcogramMax = torch.tensor(1.33)
        self.ihcogramMax = utils.comp(self.ihcogramMax, conf['scaleWeight'], conf['scaleType'])
        self.fs=16000
        
        # number of samples to be skipped due to WaveNet processing    
        self.skipLength = (2**conf['nLayers'])*conf['nStacks']      
    
        ## initialize WaveNet and load model paramaters
        self.NET = WaveNet.WaveNet(conf['nLayers'],
                                   conf['nStacks'],
                                   conf['nChannels'],
                                   conf['nResChannels'],
                                   conf['nSkipChannels'],
                                   conf['numOutputLayers'])
        
        self.NET.load_state_dict(torch.load("./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",\
                                        map_location=torch.device('cuda:0'), weights_only=True))
        
        # Freeze the IHC layers
        for param in self.NET.parameters():
            param.requires_grad= False
        #self.frame_shift = int(segdur*self.fs)
        #self.frame_len = self.frame_shift + self.skipLength
        ############################
        
        #This is the CPC style feature encoder
        self.nHiddenChannels=nHiddenUnits
        self.strides=[5, 4, 2, 2, 2]
        self.kernels=[10, 8, 4, 4, 4]
        self.nLayers=nLayers
        self.n_classes=n_classes
        
        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                #The padding for conv1d cpc is padding=stride/2, given that k is always 2*stride
                p=math.ceil(stride/2)
                #Make the convolutional layer, and initialise it well
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=p, padding_mode='zeros', bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            #Return the convolutional, dropout and activation function layer 
            return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.ReLU())
            
            
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        
        in_d=80
        for iLayer in range(self.nLayers):
            #Append the temporal convolution block
            self.conv_layers.append(
                block(
                    in_d,
                    self.nHiddenChannels,
                    self.kernels[iLayer],
                    self.strides[iLayer],
                    conv_bias=conv_bias, ) )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
                                 
            #Set the in dimension to be the number of hidden units, i.e. 80
            #After the first layer
            in_d=self.nHiddenChannels
            
         #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(self.nHiddenChannels, self.n_classes)
        
    def toString(self):
        if self.nHiddenChannels == 80:
            return "IHC_Extract"
        else:
            return "IHC_Extract_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        ##################################################################################
        # WAVENET AUDITORY TRANSDUCTION RUNNER
        ##################################################################################
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        signals=signals.permute(0, 2, 1)
        
        with torch.no_grad():
            IHC_predicted=self.NET(signals)
        
            IHC_predicted = IHC_predicted*self.ihcogramMax
            x = utils.invcomp(IHC_predicted, self.conf['scaleWeight'], self.conf['scaleType'])
        
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y

####################################################################################################################################
class SIG_Extract(nn.Module):
    """SIG extract is a neural network designed for phoneme prediction using signal as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is designed to use the IHC potentials, but altered to just use the signal alone. Based on an adapted version of CPC due to the lower inital parameters
    and comparable performance, this model has a number of improvements that should hopefully improve the model 
    performance overall. 
    
    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training.

    Next, a squeeze and excitation layer was added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, nLayers=5, conv_bias: bool = False):
        super().__init__()
            
        #This is the CPC style feature encoder
        self.nHiddenChannels=nHiddenUnits
        self.strides=[5, 4, 2, 2, 2]
        self.kernels=[10, 8, 4, 4, 4]
        self.nLayers=nLayers
        self.n_classes=n_classes
        
        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                #The padding for conv1d cpc is padding=stride/2, given that k is always 2*stride
                p=math.ceil(stride/2)
                #Make the convolutional layer, and initialise it well
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=p, padding_mode='zeros', bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            #Return the convolutional, dropout and activation function layer 
            return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.ReLU())
            
            
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        
        in_d=1
        for iLayer in range(self.nLayers):
            #Append the temporal convolution block
            self.conv_layers.append(
                block(
                    in_d,
                    self.nHiddenChannels,
                    self.kernels[iLayer],
                    self.strides[iLayer],
                    conv_bias=conv_bias, ) )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
                                 
            #Set the in dimension to be the number of hidden units, i.e. 80
            #After the first layer
            in_d=self.nHiddenChannels
            
         #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(self.nHiddenChannels, self.n_classes)
        
    def toString(self):
        if self.nHiddenChannels == 80:
            return "Extract"
        else:
            return "Extract_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Unpack my little sardines
        x, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        x=x.permute(0, 2, 1)
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y
####################################################################################################
# IHC Extract 2.0
###################################################################################################
class IHC_Extract_2(nn.Module):
    """IHC_Extract_2 is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is the second iteration of a custom built phoneme recogniser, built to use these new features. This is based on
    the Wav2vec2.0 model, where 7 layers of smaller kernels (compared to CPC) make it very slightly more lightweight than 
    the CPC model. It has a number of improvements compared to IHC_Extract and the other feature encoders tested.

    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training. Residual gradients are also used in a similar manner,
    using convolutional kernels without any non-linear activations. 

    Squeeze and excitation layers are used again here, added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, dropout: float = 0.0, mode: str = "default", \
                 required_seq_len_multiple=2, conv_bias: bool = False):
        super().__init__()
        self.n_classes=n_classes
        #######################
        #Configure the WavIHC model
        # load configuration file  
        with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml",'r') as ymlfile:
            conf = yaml.safe_load(ymlfile)   #
            self.conf=conf
            
        #Constants
        self.sigMax = torch.tensor(55)
        self.ihcogramMax = torch.tensor(1.33)
        self.ihcogramMax = utils.comp(self.ihcogramMax, conf['scaleWeight'], conf['scaleType'])
        self.fs=16000
        
        # number of samples to be skipped due to WaveNet processing    
        self.skipLength = (2**conf['nLayers'])*conf['nStacks']      
    
        ## initialize WaveNet and load model paramaters
        self.NET = WaveNet.WaveNet(conf['nLayers'],
                                   conf['nStacks'],
                                   conf['nChannels'],
                                   conf['nResChannels'],
                                   conf['nSkipChannels'],
                                   conf['numOutputLayers'])
        
        self.NET.load_state_dict(torch.load("./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",\
                                        map_location=torch.device('cuda:0'), weights_only=True))
        
        # Freeze the IHC layers
        for param in self.NET.parameters():
            param.requires_grad= False
        #self.frame_shift = int(segdur*self.fs)
        #self.frame_len = self.frame_shift + self.skipLength
        ############################
        self.nHiddenChannels=nHiddenUnits
        #This is the wav2vec2 style feature encoder
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 80
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
            
            in_d = dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(nHiddenUnits, self.n_classes)
       
    def toString(self):
        if self.nHiddenChannels == 80:
            return "IHC_Extract_2.0"
        else:
            return "IHC_Extract_2.0_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        ##################################################################################
        # WAVENET AUDITORY TRANSDUCTION RUNNER
        ##################################################################################
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        signals=signals.permute(0, 2, 1)
        
        with torch.no_grad():
            IHC_predicted=self.NET(signals)
        
            IHC_predicted = IHC_predicted*self.ihcogramMax
            x = utils.invcomp(IHC_predicted, self.conf['scaleWeight'], self.conf['scaleType'])
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y

#########################################################################################################################################################################################
# IHC Extract3
########################################################################################################################################################################################
class IHC_Extract_3(nn.Module):
    """IHC_Extract_3 is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is the second iteration of a custom built phoneme recogniser, built to use these new features. This is based on
    the Wav2vec2.0 model, where 7 layers of smaller kernels (compared to CPC) make it very slightly more lightweight than 
    the CPC model. It has a number of improvements compared to IHC_Extract and the other feature encoders tested.

    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training. Residual gradients are also used in a similar manner,
    using convolutional kernels without any non-linear activations. 

    Squeeze and excitation layers are used again here, added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    The difference between this and IHC_Extract_2.0 is that for this model, the convolutions are depthwise, i.e. there is a filter for each IHC channel 
    and they are not mixed until the end.
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, dropout: float = 0.0, mode: str = "default", \
                 required_seq_len_multiple=2, conv_bias: bool = False):
        super().__init__()
        self.n_classes=n_classes
        #######################
        #Configure the WavIHC model
        # load configuration file  
        with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml",'r') as ymlfile:
            conf = yaml.safe_load(ymlfile)   #
            self.conf=conf
            
        #Constants
        self.sigMax = torch.tensor(55)
        self.ihcogramMax = torch.tensor(1.33)
        self.ihcogramMax = utils.comp(self.ihcogramMax, conf['scaleWeight'], conf['scaleType'])
        self.fs=16000
        
        # number of samples to be skipped due to WaveNet processing    
        self.skipLength = (2**conf['nLayers'])*conf['nStacks']      
    
        ## initialize WaveNet and load model paramaters
        self.NET = WaveNet.WaveNet(conf['nLayers'],
                                   conf['nStacks'],
                                   conf['nChannels'],
                                   conf['nResChannels'],
                                   conf['nSkipChannels'],
                                   conf['numOutputLayers'])
        
        self.NET.load_state_dict(torch.load("./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",\
                                        map_location=torch.device('cuda:0'), weights_only=True))
        
        # Freeze the IHC layers
        for param in self.NET.parameters():
            param.requires_grad= False
        #self.frame_shift = int(segdur*self.fs)
        #self.frame_len = self.frame_shift + self.skipLength
        ############################
        self.nHiddenChannels=nHiddenUnits
        #This is the wav2vec2 style feature encoder
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                #This creates depthwise convolutions (groups = n_in)
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, groups=n_in)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 80
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
            
            in_d = dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(nHiddenUnits, self.n_classes)
       
    def toString(self):
        if self.nHiddenChannels == 80:
            return "IHC_Extract_3.0"
        else:
            return "IHC_Extract_3.0_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        ##################################################################################
        # WAVENET AUDITORY TRANSDUCTION RUNNER
        ##################################################################################
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        signals=signals.permute(0, 2, 1)
        
        with torch.no_grad():
            IHC_predicted=self.NET(signals)
        
            IHC_predicted = IHC_predicted*self.ihcogramMax
            x = utils.invcomp(IHC_predicted, self.conf['scaleWeight'], self.conf['scaleType'])
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y
        
##############################################################################################################################################################################################
# SIG Extract2
##########################################################################################################################################################################################
class SIG_Extract_2(nn.Module):
    """SIG_Extract_2 is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is the second iteration of a custom built phoneme recogniser, built to use these new features. This is based on
    the Wav2vec2.0 model, where 7 layers of smaller kernels (compared to CPC) make it very slightly more lightweight than 
    the CPC model. It has a number of improvements compared to IHC_Extract and the other feature encoders tested.

    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no explicit methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training. Residual gradients are also used in a similar manner,
    using convolutional kernels without any non-linear activations. 

    Squeeze and excitation layers are used again here, added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, dropout: float = 0.0, mode: str = "default", \
                 required_seq_len_multiple=2, conv_bias: bool = False):
        super().__init__()
        self.n_classes=n_classes
        
        self.nHiddenChannels=nHiddenUnits
        #This is the wav2vec2 style feature encoder
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 1
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
            
            in_d = dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(nHiddenUnits, self.n_classes)
       
    def toString(self):
        if self.nHiddenChannels == 80:
            return "SIG_Extract_2.0"
        else:
            return "SIG_Extract_2.0_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        x=signals.permute(0, 2, 1)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y

##############################################################################################################################################################################################
# SIG Extract3
##########################################################################################################################################################################################
class SIG_Extract_3(nn.Module):
    """SIG_Extract_3 is a neural network designed for phoneme prediction using predicted IHC potentials as input
    The goal is to understand how discriminative IHC potentials are as a possible alternative to common signal
    alone or signal-scaling techniques. This works on the assumption that linear separability defines the 
    accessibility of information for downstream tasks. The major limitation of this model is the lack of context -
    speech features depend on context and have both short- and long-term dependencies. 
    
    This is the second iteration of a custom built phoneme recogniser, built to use these new features. This is based on
    the Wav2vec2.0 model, where 7 layers of smaller kernels (compared to CPC) make it very slightly more lightweight than 
    the CPC model. It has a number of improvements compared to IHC_Extract and the other feature encoders tested.

    First, increased stability. Evident from IHC_Cpc and especially IHC_Wav2vec2.0 was that 
    the models had no explicit methods of stabilising learning. Here, batch normalisation was used to reduce the unwanted 
    stochastic effects of training, and to help prevent overfitting. This is used instead of dropout in this case.
    An added bonus is that this may also speed up training. Residual gradients are also used in a similar manner,
    using convolutional kernels without any non-linear activations. 

    Squeeze and excitation layers are used again here, added at every stage. This was to explictly model the interdependencies between
    neural network channels - as a form of channel attention. This is highly applicable to this use case, as the channels are
    meaningful representations of the sound. This uses the 80 hidden channels.

    The difference between this and the Extract_2.0 system is that for this model, the convolutions are depthwise, i.e. there is a filter for each channel 
    and they are not mixed until the end.
    
    """
    
    def __init__(self, config=None, mlp_config=None, n_classes=39, device='cuda:0',\
                 nHiddenUnits=80, dropout: float = 0.0, mode: str = "default", \
                 required_seq_len_multiple=2, conv_bias: bool = False):
        super().__init__()
        self.n_classes=n_classes
        
        self.nHiddenChannels=nHiddenUnits
        #This is the wav2vec2 style feature encoder
        conv_layers=[(nHiddenUnits, 10, 5)] + [(nHiddenUnits, 3, 2)] * 4 + [(nHiddenUnits,2,2)] + [(nHiddenUnits,2,2)]
        
        #Make sure the mode is one of the two allowed
        assert mode in {"default", "layer_norm"}

        #Create a temporal convolution block object
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            #This can make convolutions, and initialise them to the kaiming method
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, groups=n_in)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(n_out),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.BatchNorm1d(n_out), nn.GELU())
            
        #This hard codes the fact that the input dimension (i.e. channel) is 1 at the start
        in_d = 1
        
        #Create a temporal convolution list
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

            #Add a squeeze and excitation layer for every regular convolutional layer
            self.conv_layers.append(SEBlock(self.nHiddenChannels))
            
            in_d = dim
            
        #The final fully connected softmax layer for seq2seq prediction
        self.fc = nn.Linear(nHiddenUnits, self.n_classes)
       
    def toString(self):
        if self.nHiddenChannels == 80:
            return "SIG_Extract_3.0"
        else:
            return "SIG_Extract_3.0_{}".format(self.nHiddenChannels)
            
    def forward(self, inputs):
        """Forward function for both training and testing.

        input:
          inputs: (batch_size, seq_len, mel_dim)
          lengths: (batch_size,)

        return:
          predicted_phone: (batch_size, n_classes, seq_len)
        """
        ###################################################
        # Feature Encoder
        ###################################################
        #Give our inputs to WavIHC to get the predicted IHC potentials
        #Unpack my little sardines
        signals, self.lens = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        sigLen = signals.shape[1]
        
        #Change the format from (Batch, Len, Num Channels) to (Batch, Num Channels, Len)
        x=signals.permute(0, 2, 1)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
        #Change the last dimension to be the channel
        x=x.permute(0, 2, 1)
        
        #Pass it through the classification layer
        y=self.fc(x)
        
        y=y.permute(0, 2, 1)
        
        return y