import os
import csv
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import numpy as np

def count_audio_by_gender(protocol_path):
    male_count = 0
    female_count = 0
    mixed_count = 0
    with open(protocol_path, 'r') as file:
        for line in file:
            speaker, gender,  file_name, label = line.split()
            if gender == 'male':
                male_count += 1 
            elif gender == 'female':
                female_count += 1
            else:
                mixed_count += 1
    return male_count, female_count, mixed_count


def split_meta_by_speaker(meta_path, save_dir):
    metadata_df = pd.read_csv(meta_path, sep=' ', header=None, names = ['speaker', 'gender', 'file_name', 'label'])
    unique_speakers = metadata_df['speaker'].unique()
    train_speakers, test_speakers = train_test_split(unique_speakers, test_size=0.5, random_state=4321)
    dev_speakers, eval_speakers = train_test_split(test_speakers, test_size=0.5, random_state=4321)
    
    train_df = metadata_df[metadata_df['speaker'].isin(train_speakers)]
    dev_df = metadata_df[metadata_df['speaker'].isin(dev_speakers)]
    eval_df = metadata_df[metadata_df['speaker'].isin(eval_speakers)]
    
    train_df.to_csv(os.path.join(save_dir, 'train_meta.txt'), sep=' ', index=False, header=False)
    dev_df.to_csv(os.path.join(save_dir, 'dev_meta.txt'), sep=' ', index=False, header=False)
    eval_df.to_csv(os.path.join(save_dir, 'eval_meta.txt'), sep=' ', index=False, header=False)
    return len(train_df), len(dev_df), len(eval_df)


def get_meta_csv_paths(base_dir):
    meta_csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "meta.csv":
                meta_csv_paths.append(os.path.join(root, file))
    return meta_csv_paths

def print_speakers_info(meta_path):
    speakers_info = {}
    with open(meta_path, 'r') as file:
        for line in file:
            speaker, gender, file_name, label = line.strip().split()
            if speaker in speakers_info:
                speakers_info[speaker]['file_count'] += 1
            else:
                speakers_info[speaker] = {'gender': gender, 'file_count': 1}
    for speaker, info in speakers_info.items():
        print(f"Speaker: {speaker}, Gender: {info['gender']}, Files: {info['file_count']}")

def create_new_meta(meta_dir, speakers):
    meta_path = os.path.join(meta_dir, 'meta.txt')
    with open(meta_path , 'w') as file:
        for speaker, speaker_data in speakers.items():
            for gender, file_name, label in speaker_data:
                file.write(f"{speaker} {gender} {file_name} {label}\n")

            print(f"Speaker: {speaker} - Number of files: {len(speaker_data)}")
        file.close()  
    print(f"Total number of speakers: {len(speakers)}")



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

       


def copy_files_to_new_dir(meta_csv_paths, concat_dataset_dir):
    meta_cvs_counter = 0
    file_counter = 1
    speakers = {}

    for meta_csv_path in meta_csv_paths:
        with open(meta_csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
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
                    
                # TODO SOLVE THE WEIRDNESS WITH ITALIAN NOVELLE
                    

                #speaker = f"{speaker_id}_male" if gender == 'male' else f"{speaker_id}_female"

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

                if gender == 'male':
                    spoof_chunks_list = split_audio_into_chunks(spoof_source_path, audio_file_dest_path, file_counter)
                    file_counter += len(spoof_chunks_list)
                    for chunk in spoof_chunks_list:
                        speakers[speaker].append((gender, chunk, 'spoof'))
                    orig_chunks_list = split_audio_into_chunks(orig_source_path, audio_file_dest_path, file_counter)
                    file_counter += len(orig_chunks_list)
                    for chunk in orig_chunks_list:
                        speakers[speaker].append((gender, chunk, 'bonafide'))
                else:
                    # For female speakers, copy files without splitting
                    new_filename_s = f"{file_counter}_{os.path.basename(path_s)}"
                    new_filename_o = f"{file_counter+1}_{os.path.basename(path_o)}"
                    spoof_dest_path = os.path.join(audio_file_dest_path, new_filename_s)
                    orig_dest_path = os.path.join(audio_file_dest_path, new_filename_o)
                    shutil.copy(spoof_source_path, spoof_dest_path)
                    shutil.copy(orig_source_path, orig_dest_path)
                    file_counter += 2
                    speakers[speaker].append((gender, new_filename_s, 'spoof'))
                    speakers[speaker].append((gender, new_filename_o, 'bonafide'))
            #print(speakers)
                
        meta_cvs_counter += 1
        print(f"Processed {meta_cvs_counter} meta.csv files out of {len(meta_csv_paths)}")
        """if meta_cvs_counter == 2:
            return speakers"""
    return speakers


            




dataset_dir = 'database/MLAAD_GB'

if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
mlaand_v2_base_dir = r"database\mlaad_v2\MLAAD\fake"


#meta_csv_paths = get_meta_csv_paths(mlaand_v2_base_dir)
#print(len(meta_csv_paths))
#speakers_dict = copy_files_to_new_dir(meta_csv_paths, dataset_dir)
#create_new_meta(dataset_dir, speakers_dict)




male, female, mix = count_audio_by_gender(dataset_dir+'/meta.txt')
print(f"Male audio: {male}\nFemale audio: {female}\nMixed audio: {mix}")

len_train, len_dev, len_eval = split_meta_by_speaker(os.path.join(dataset_dir, 'meta.txt'), dataset_dir)
print(f"Train: {len_train}\nDev: {len_dev}\nEval: {len_eval}")
print()
print_speakers_info(os.path.join(dataset_dir, 'meta.txt'))

"""trial_meta={}
file_name_list_trial=[]
with open('keys/CM/trial_metadata.txt', 'r') as file:
    for line in file:
        _, file_name, _, _, _, _, _, _ = line.split()
        file_name_list_trial.append(file_name)

file_name_list_eval=[]
with open('MLAAD_on_MLAAD_eval.txt', 'r') as file2:
    for line in file2:
        file_name, _= line.split()
        file_name_list_eval.append(file_name)
print(len(file_name_list_trial), len(file_name_list_eval))

set_a = set(file_name_list_trial)
set_b = set(file_name_list_eval)

only_in_a = set_a - set_b
only_in_b = set_b - set_a
common = set_a & set_b

print(f"Only in list A: {only_in_a}")
print(f"Only in list B: {only_in_b}")
#print(f"Common items: {common}")
file_path = 'MLAAD_on_MLAAD_eval.txt'
temp_file_path = 'temp.txt'
# Open the original file in read mode and a temporary file in write mode
with open(file_path, 'r') as read_file, open(temp_file_path, 'w') as write_file:
    # Iterate over each line in the original file
    for line in read_file:
        # If the line contains any of the items to keep, write it to the temp file
        if any(item_to_keep in line for item_to_keep in file_name_list_trial):
            write_file.write(line)

# Replace the original file with the temporary file
import os
os.replace(temp_file_path, file_path)"""