import os
import csv
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import numpy as np




"""
In order for the script to work, the following directory structure is required:
% database_path/
    %   |- DF
    %      |- ...
    %   |- ITW           <- The original ITW dataset link is provided in the instructions.txt file
    %      |- wav 
    %   |- mlaad_v2      <- The original MLAAD dataset link is provided in the instructions.txt file 
    %      |- MLAAD
    %         |- fake
    %            |- ar
    %            |- bg
    %            |- cs
    %            |- ...
    %   | - MAILABS      <- The original M-AILABS dataset link is provided in the instructions.txt file
    %      |- de_DE
    %      |- en_UK
    %      |- en_US
    %      |- ...

"""


def create_demo_dataset(meta_path, dataset_dir, dest_dataset, dataset_name, sample_size=100):
    
    if dataset_name == 'MLAAD_GB':    
        metadata_df = pd.read_csv(meta_path, sep=' ', header=None, names = ['speaker', 'language', 'gender', 'file_name', 'label'])
        file_col = 'file_name'
    elif dataset_name == 'ITW':
        metadata_df = pd.read_csv(meta_path, header=0, names = ['file', 'speaker', 'label'])
        file_col = 'file'
    
    os.makedirs(dest_dataset, exist_ok=True)

    sample_df = metadata_df.sample(n=sample_size, random_state=42)
    for filename in sample_df[file_col]:
        source_path = os.path.join(dataset_dir, 'wav', filename)
        wav_dir = os.path.join(dest_dataset, 'wav')
        os.makedirs(wav_dir, exist_ok=True)
        dest_path = os.path.join(wav_dir, filename)
        shutil.copy(source_path, dest_path)
    if dataset_name == 'MLAAD_GB':
        new_meta_path = os.path.join(dest_dataset, 'meta.txt')
        sample_df.to_csv(new_meta_path, sep=' ', index=False, header=False)
    elif dataset_name == 'ITW':
        new_meta_path = os.path.join(dest_dataset, 'meta.csv')
        sample_df.to_csv(new_meta_path, index=False, header=True)
    
    

    


def split_meta_by_speaker(meta_path, save_dir, dataset_name, train_size, dev_size, eval_size, rand_seed):
    '''Splits the metadata file by speaker into train, dev and eval sets and saves them to the specified directory.'''
    if train_size + dev_size + eval_size != 1:
        raise ValueError("The sum of train, dev and eval sizes must be equal to 1")
    if dataset_name == 'MLAAD_GB':    
        metadata_df = pd.read_csv(meta_path, sep=' ', header=None, names = ['speaker', 'language', 'gender', 'file_name', 'label'])
    elif dataset_name == 'ITW':
        metadata_df = pd.read_csv(meta_path, header=0, names = ['file', 'speaker', 'label'])
    unique_speakers = metadata_df['speaker'].unique()
    train_speakers, test_speakers = train_test_split(unique_speakers, test_size=1 - train_size, random_state=rand_seed)
    dev_speakers, eval_speakers = train_test_split(test_speakers, test_size=eval_size / (1 - train_size), random_state=rand_seed)
    
    train_df = metadata_df[metadata_df['speaker'].isin(train_speakers)]
    dev_df = metadata_df[metadata_df['speaker'].isin(dev_speakers)]
    eval_df = metadata_df[metadata_df['speaker'].isin(eval_speakers)]
    
    train_df.to_csv(os.path.join(save_dir, 'train_meta.txt'), sep=' ', index=False, header=False)
    dev_df.to_csv(os.path.join(save_dir, 'dev_meta.txt'), sep=' ', index=False, header=False)
    eval_df.to_csv(os.path.join(save_dir, 'eval_meta.txt'), sep=' ', index=False, header=False)

    return len(train_df), len(dev_df), len(eval_df)


def get_meta_csv_paths(base_dir):
    '''Returns a list of paths to all meta.csv files in the specified directory and its subdirectories.
    It is used to get the paths of all meta.csv files in the MLAAD dataset.'''
    meta_csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "meta.csv":
                meta_csv_paths.append(os.path.join(root, file))
    return meta_csv_paths



def create_new_meta_mailabs_and_mlaad(meta_dir, speakers):
    '''Creates a new meta file for the concatenated M-AILABS and MLAAD datasets.'''
    meta_path = os.path.join(meta_dir, 'meta.txt')
    with open(meta_path , 'w') as file:
        for speaker, speaker_data in speakers.items():
            for gender, language, file_name, label in speaker_data:
                file.write(f"{speaker} {language} {gender} {file_name} {label}\n")

            #print(f"Speaker: {speaker} - Number of files: {len(speaker_data)}")
        file.close()  
    #print(f"Total number of speakers: {len(speakers)}")



def split_audio_into_chunks(filename, destination_dir, file_counter, chunk_length_sec=5.0):
    """
    Splits an audio file into chunks of a specified length and saves them to a specified directory.
    """
    audio, sr = librosa.load(filename, sr=None)  # Load with original sample rate
    num_samples_per_chunk = int(chunk_length_sec * sr)
    total_chunks = np.ceil(len(audio) / num_samples_per_chunk).astype(int)
    chunks_list = []
    for i in range(total_chunks):       
        start_sample = i * num_samples_per_chunk
        end_sample = start_sample + num_samples_per_chunk
        chunk = audio[start_sample:end_sample]
        chunk_filename = f"{file_counter+i}_{os.path.basename(filename)}"
        chunks_list.append(chunk_filename)
        chunk_file_path = os.path.join(destination_dir, chunk_filename)
        sf.write(chunk_file_path, chunk, sr)

    return chunks_list

def create_MLAAD_GB(meta_csv_paths, concat_dataset_dir):
    '''Creates the MLAAD_GB dataset by concatenating the MLAAD dataset and the M-AILABS dataset.'''
    meta_cvs_counter = 0
    file_counter = 1
    speakers = {}

    for meta_csv_path in meta_csv_paths:
        with open(meta_csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                path, path_o, _, _, _, _, _, _, _ = row[0].split('|')
                if path_o.split('/')[0] == 'fr_FR':
                    gender = path_o.split('/')[1]
                    speaker = path_o.split('/')[2]
                else:
                    gender = path_o.split('/')[2]
                    if path_o.split('/')[3].startswith('novelle_per_un_anno'):
                        speaker = path_o.split('/')[3][:-3]
                    else:
                        speaker = path_o.split('/')[3]

                # Keep track of the original and spoofed languages in ISO 639-1 format
                original_language = path_o.split('/')[0][:2]
                spoofed_language = row[0].split('|')[2]

                # Keep track of speakers and their files
                if speaker not in speakers:
                    speakers[speaker] = []

                # Adjusted paths to reflect actual file locations
                path_s = path[1:]
                path_o = path_o  # Assuming path_o is correct

                spoof_source_path = os.path.join('database/mlaad_v2/MLAAD'+ path_s)
                orig_source_path = os.path.join('database/MAILABS/'+ path_o)
                audio_file_dest_path = os.path.join(concat_dataset_dir, 'wav')
                os.makedirs(audio_file_dest_path, exist_ok=True)

                # Split the audio files of male speakers into chunks of 5 seconds
                if gender == 'male':
                    spoof_chunks_list = split_audio_into_chunks(spoof_source_path, audio_file_dest_path, file_counter)
                    file_counter += len(spoof_chunks_list)
                    for chunk in spoof_chunks_list:
                        speakers[speaker].append((gender, original_language, chunk, 'spoof'))
                    orig_chunks_list = split_audio_into_chunks(orig_source_path, audio_file_dest_path, file_counter)
                    file_counter += len(orig_chunks_list)
                    for chunk in orig_chunks_list:
                        speakers[speaker].append((gender, spoofed_language, chunk, 'bonafide'))
                else:
                    # For female speakers, copy files without splitting
                    new_filename_s = f"{file_counter}_{os.path.basename(path_s)}"
                    new_filename_o = f"{file_counter+1}_{os.path.basename(path_o)}"
                    spoof_dest_path = os.path.join(audio_file_dest_path, new_filename_s)
                    orig_dest_path = os.path.join(audio_file_dest_path, new_filename_o)
                    shutil.copy(spoof_source_path, spoof_dest_path)
                    shutil.copy(orig_source_path, orig_dest_path)
                    file_counter += 2
                    speakers[speaker].append((gender, original_language, new_filename_s, 'spoof'))
                    speakers[speaker].append((gender, spoofed_language, new_filename_o, 'bonafide'))
               
        meta_cvs_counter += 1
        print(f"Processed {meta_cvs_counter} meta.csv files out of {len(meta_csv_paths)}")
    create_new_meta_mailabs_and_mlaad(concat_dataset_dir, speakers)


if __name__ == '__main__':

    ### Splitthe ITW meta file based on speakers ###
    len_train, len_dev, len_eval = split_meta_by_speaker('database/ITW/meta.csv', 'database/ITW_1', dataset_name='ITW', train_size=0.5, dev_size=0.3, eval_size=0.2, rand_seed=123)
    print(f"\nITW\nTrain: {len_train}\nDev: {len_dev}\nEval: {len_eval}")



    ### Destination directory for the concatenated M-AILABS and MLAAD datasets ###
    mlaad_gb_dataset_dir = 'database/MLAAD_GB_v3'

    if not os.path.exists(mlaad_gb_dataset_dir):
            os.mkdir(mlaad_gb_dataset_dir)

    mlaand_v2_base_dir = r"database\mlaad_v2\MLAAD\fake"

    mlaad_meta_csv_paths = get_meta_csv_paths(mlaand_v2_base_dir)
    create_MLAAD_GB(mlaad_meta_csv_paths, mlaad_gb_dataset_dir)
    
    # Split the demo MLAAD_GB dataset by speaker
    len_train, len_dev, len_eval = split_meta_by_speaker(meta_path=os.path.join(mlaad_gb_dataset_dir, 'meta.txt'), save_dir=mlaad_gb_dataset_dir, dataset_name='MLAAD_GB', train_size=0.5, dev_size=0.3, eval_size=0.2, rand_seed=42)
    print(f"\nMLAAD_GB_demo\nTrain: {len_train}\nDev: {len_dev}\nEval: {len_eval}")

    ################################################################################################
    ### To create demo dataset for ITW and MLAAD_GB datasets, uncomment the following code block ###
    ### and run the script with the required dataset directory.                                  ###
    ################################################################################################
    '''    
    ### Create the demo ITW dataset ###
    itw_dataset_dir = 'database/ITW'
    itw_dataset_dir_demo = 'database/ITW_demo' 
    create_demo_dataset(os.path.join(itw_dataset_dir, 'meta.csv'), itw_dataset_dir, itw_dataset_dir_demo, 'ITW')

    # Split the demo ITW dataset by speaker
    len_train, len_dev, len_eval = split_meta_by_speaker(meta_path=itw_dataset_dir_demo+'/meta.csv', save_dir=itw_dataset_dir_demo, dataset_name='ITW', train_size=0.5, dev_size=0.3, eval_size=0.2, rand_seed=123)
    print(f"\nITW_demo\nTrain: {len_train}\nDev: {len_dev}\nEval: {len_eval}")

    


    ### Create the demo MLAAD_GB dataset ###
    mlaad_gb_dataset_dir = 'database/MLAAD_GB_v2'
    mlaad_gb_dataset_dir_demo = 'database/MLAAD_GB_demo'
    create_demo_dataset(os.path.join(mlaad_gb_dataset_dir, 'meta.txt'), mlaad_gb_dataset_dir, mlaad_gb_dataset_dir_demo, 'MLAAD_GB', sample_size=100)

    # Split the demo MLAAD_GB dataset by speaker
    len_train, len_dev, len_eval = split_meta_by_speaker(meta_path=os.path.join(mlaad_gb_dataset_dir_demo, 'meta.txt'), save_dir=mlaad_gb_dataset_dir_demo, dataset_name='MLAAD_GB', train_size=0.5, dev_size=0.3, eval_size=0.2, rand_seed=42)
    print(f"\nMLAAD_GB_demo\nTrain: {len_train}\nDev: {len_dev}\nEval: {len_eval}")
    
    '''


