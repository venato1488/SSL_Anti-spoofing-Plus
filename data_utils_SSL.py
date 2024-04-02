import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random
import csv


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

####################################
## Dataset processing and loading ##
####################################





def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    """
    Generate a list of spoof and bonafide files files based on the provided metadata file and the parameter provided at the execution
    of the main_SSL_LA/DF. Key is the file name and value is the label (1 for bonafide, 0 for spoof).

    Args:
        dir_meta (str): The path to the metadata file.
        is_train (bool, optional): Specifies whether the function is used for training. Defaults to False.
        is_eval (bool, optional): Specifies whether the function is used for evaluation. Defaults to False.

    Returns:
        dict or list: If is_train is True, returns a dictionary containing the metadata for each file and a list of file names.
                      If is_eval is True, returns a list of file names.
                      Otherwise, returns a dictionary containing the metadata for each file and a list of file names.
    """
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list



import numpy as np
def pad(x, max_len=64600):
    """
    This function applies a repeat-padding strategy to ensure audio samples reach a uniform 
    length of 64,600 samples, suitable for datasets with variable-length entries. 
    If an audio sample is already at or above this length, it's truncated to fit precisely. 
    For shorter samples, the function calculates the necessary number of repeats to reach or 
    exceed the target length, uses np.tile to replicate the audio accordingly, and then trims 
    the result to ensure it matches the desired length exactly. This approach maintains the 
    audio's original content without introducing silence, though it may lead to repetition artifacts.

    Parameters:
    - x (numpy.ndarray): The input array to be padded.
    - max_len (int): The maximum length to pad the array to. Default is 64600.

    Returns:
    - numpy.ndarray: The padded array.
    """

    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
			

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
           return len(self.list_IDs)


    def __getitem__(self, index):
            """
            Retrieves a single item from the dataset at the specified index. It loads an audio file
            using librosa, applies preprocessing using the RawBoost algorithm and data augmentation,
            and pads the audio file to a fixed length of 64600 samples (~ 4 sec audio).
            Crucial for training the model as it ensures that all input arrays have the same length.
            
            Parameters:
            - index (int): The index of the item to retrieve from the dataset.

            Returns:
            - x_inp (Tensor): The preprocessed and padded audio file.
            - target (int): The label of the audio file. 1 for bonafide, 0 for spoof.
            """
            utt_id = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
            return x_inp, target
            
           
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''
               
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id  


###############################
#### ITW and MLAAD Dataset ####
###############################



def evaluation_file_creator(metadata_dict):
    """
    Creates an evaluation file formated in the same way ASVspoof 2021 DF evaluation file is formatted.
    """
    keys_path = 'keys/CM/trial_metadata.txt'
    with open(keys_path, 'w') as file:
        for key, label in metadata_dict.items():
            file.write('- {} - - - {} - eval\n'.format(key, label))
        file.close()
        print("Keys file created successfully")

def parse_line_simple(line):
    # Split the line into three parts, but only split at the first two spaces found
    parts = line.split(maxsplit=2)
    if len(parts) != 3:
        raise ValueError("Line format is incorrect")

    key = parts[0]
    # Assuming the name is always in quotes, strip them off
    name = parts[1].strip('\"')
    label = parts[2]

    return key, name, label


def genSpoof_list_ITW(metadata_file_path, is_train=False, is_eval=False):
    """
    Generate a list of spoof and bonafide files files based on the provided metadata file and the parameter provided at the execution
    of the main_SSL_LA/DF. Key is the file name and value is the label (1 for bonafide, 0 for spoof).

    Args:
        dir_meta (str): The path to the metadata file.
        is_train (bool, optional): Specifies whether the function is used for training. Defaults to False.
        is_eval (bool, optional): Specifies whether the function is used for evaluation. Defaults to False.

    Returns:
        dict or list: If is_train is True, returns a dictionary containing the metadata for each file and a list of file names.
                      If is_eval is True, returns a list of file names.
                      Otherwise, returns a dictionary containing the metadata for each file and a list of file names.
    """
    d_meta = {}
    file_list=[]
    with open(metadata_file_path, 'r') as f:
        l_meta = f.readlines()
        if (is_train):
            for line in l_meta:
                key, _,label = parse_line_simple(line)
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bona-fide' else 0
            return d_meta,file_list
        
        elif(is_eval):
            metadata_dict_for_keys = {}
            for line in l_meta:              
                key, _, label = parse_line_simple(line)
                metadata_dict_for_keys[key] = 'bonafide' if label == 'bona-fide' else 'spoof'
                file_list.append(key)
            evaluation_file_creator(metadata_dict_for_keys)
            return file_list
        else:
            for line in l_meta:
                key, _,label = parse_line_simple(line)
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bona-fide' else 0
            return d_meta,file_list

def genSpoof_list_MLAAD(metadata_dir, is_train=False, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(metadata_dir, 'r') as f:
        l_meta = f.readlines()

        if (is_train):
            for line in l_meta:
                _,_,key,label = line.strip().split()
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
            return d_meta,file_list
        
        elif(is_eval):
            metadata_dict_for_keys = {}
            for line in l_meta:
                _,_,key,label = line.strip().split()
                metadata_dict_for_keys[key] = 'bonafide' if label == 'bonafide' else 'spoof'
                file_list.append(key)
            evaluation_file_creator(metadata_dict_for_keys)
            return file_list
        else:
            for line in l_meta:
                _,_,key,label = line.strip().split()
                
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list




class Wav_Containing_Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples) 
    
    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
                
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+utt_id, sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id
        


class Wav_Containing_Dataset_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
           return len(self.list_IDs)


    def __getitem__(self, index):
            """
            Retrieves a single item from the dataset at the specified index. It loads an audio file
            using librosa, applies preprocessing using the RawBoost algorithm and data augmentation,
            and pads the audio file to a fixed length of 64600 samples (~ 4 sec audio).
            Crucial for training the model as it ensures that all input arrays have the same length.
            
            Parameters:
            - index (int): The index of the item to retrieve from the dataset.

            Returns:
            - x_inp (Tensor): The preprocessed and padded audio file.
            - target (int): The label of the audio file. 1 for bonafide, 0 for spoof.
            """
            utt_id = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+utt_id, sr=16000) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
            return x_inp, target

#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    """
    Applies data augmentation techniques to the audio features as part of the preprocessing pipeline.
    It supports three different algorithms: Convolutive noise, Impulsive noise, and Coloured additive noise,
    either individually or in combination. This preprocessing step is crucial for enhancing the robustness
    and generalisation of the model trained by simulating different types of noise and distortions that
    might be present in the real-world data.
    
    Parameters:
    - feature (numpy.ndarray): The input audio feature to be processed.
    - sr (int): The sample rate of the audio feature. Default is 16000.
    - args (Namespace): The arguments passed to the main_SSL_LA/DF. It contains the parameters for the RawBoost algorithms.
                        By default, for the Convolutive noise and ISD additive noise algorithms in series.
    - algo (int): The algorithm to be used for data augmentation. It can be 1, 2, 3, 4, 5, 6, 7, or 8.
    
    Returns:
    - feature (numpy.ndarray): The preprocessed audio feature after applying the RawBoost algorithm.
    """
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
