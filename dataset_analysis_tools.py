import pandas as pd
import os
import wave
import csv


def read_meta(meta_csv_paths):
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
                if gender not in speakers:
                    speakers[gender] = []
                
                if gender == 'male':
                    if speaker not in speakers['male']:
                        speakers['male'].append(speaker)
                elif gender == 'female':
                    if speaker not in speakers['female']:
                        speakers['female'].append(speaker)
                elif gender == 'mix':
                    if speaker not in speakers['mix']:
                        speakers['mix'].append(speaker)
    return speakers


def get_meta_csv_paths(base_dir):
    meta_csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "meta.csv":
                meta_csv_paths.append(os.path.join(root, file))
    return meta_csv_paths

def count_audio_by_gender(protocol_path):
    male_count = 0
    female_count = 0
    mixed_count = 0
    languages_count = {}
    with open(protocol_path, 'r') as file:
        for line in file:
            speaker, language, gender, file_name, label = line.split()
            if gender == 'male':
                male_count += 1 
            elif gender == 'female':
                female_count += 1
            else:
                mixed_count += 1
            if language in languages_count:
                languages_count[language] += 1
            else:
                languages_count[language] = 1
    return male_count, female_count, mixed_count, languages_count

def calculate_total_duration(directory):
    total_duration = 0.0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                filepath = os.path.join(dirpath, filename)
                # Skip if it is symbolic link
                if not os.path.islink(filepath):
                    with wave.open(filepath, 'rb') as audio_file:
                        frames = audio_file.getnframes()
                        rate = audio_file.getframerate()
                        duration = frames / float(rate)
                        total_duration += duration
    return total_duration

def print_speakers_info_mlaad(meta_path):
    speakers_info = {}
    with open(meta_path, 'r') as file:
        for line in file:
            speaker, language, gender, file_name, label = line.strip().split()
            if speaker in speakers_info:
                speakers_info[speaker]['file_count'] += 1
            else:
                speakers_info[speaker] = {'gender': gender, 'file_count': 1}
    for speaker, info in speakers_info.items():
        print(f"Speaker: {speaker}, Gender: {info['gender']}, Files: {info['file_count']}")


def count_word_occurrences(file_path):
    spoof_count = 0
    bonafide_count = 0

    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        if 'spoof' in row.values:
            spoof_count += 1
        if 'bona-fide' in row.values:
            bonafide_count += 1

    return spoof_count, bonafide_count


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

def count_labels(metadata_file_path, dataset_name):
    spoof_count = 0
    bonafide_count = 0

    with open(metadata_file_path, 'r') as f:
        l_meta = f.readlines()
        if dataset_name == 'ITW':
            for line in l_meta:
                _,_,label = parse_line_simple(line)           
                if 'spoof' in label:
                    spoof_count += 1
                elif 'bona-fide' in label:
                    bonafide_count += 1
        elif dataset_name == 'MLAAD':
            for line in l_meta:
                _,_,_,_,label = line.strip().split()
                if label == 'spoof':
                    spoof_count += 1
                elif label == 'bonafide':
                    bonafide_count += 1
        return spoof_count, bonafide_count


def print_unique_names(file_path):
    names=[]
    df = pd.read_csv(file_path)
    unique_names = df.iloc[:, 1].unique()  # Include the first row when extracting unique names
    for name in unique_names:
        #print(name)
        names.append(name)
    
    return names

def count_name_occurrences(file_path, names):
    name_counts = {}

    df = pd.read_csv(file_path)
    for name in names:
        count = df[df.iloc[:, 1] == name].shape[0]
        name_counts[name] = count

    return name_counts

# Calculate the duration of audio files in the MLAAD dataset based on the language
path_list = get_meta_csv_paths(r'database\mlaad_v2\MLAAD\fake')
lang_dur = {}
for meta in path_list:
    language = meta.split('\\')[4]
    if language not in lang_dur:
        lang_dur[language] = 0
        lang_dur[language] += calculate_total_duration(os.path.dirname(meta))
    else:
        lang_dur[language] += calculate_total_duration(os.path.dirname(meta))
for language, total_duration in lang_dur.items():
    print(f"Duration of files in {language}: {total_duration/3600} hours")

# Count the number of audio files training, validation,and evaluation partitions of MLAAD and ITW datasets
itw_path = 'database//ITW/'
mlaad_path = 'database//MLAAD_GB_v2/'


itw_train_spooof, itw_train_bonafide = count_labels(itw_path + 'train_meta.txt', 'ITW')
itw_dev_spooof, itw_dev_bonafide = count_labels(itw_path + 'dev_meta.txt', 'ITW')
itw_eval_spooof, itw_eval_bonafide = count_labels(itw_path + 'eval_meta.txt', 'ITW')
print(f"ITW Train: Spoof: {itw_train_spooof}, Bonafide: {itw_train_bonafide}")
print(f"ITW Dev: Spoof: {itw_dev_spooof}, Bonafide: {itw_dev_bonafide}")
print(f"ITW Eval: Spoof: {itw_eval_spooof}, Bonafide: {itw_eval_bonafide}")

mlaad_train_spooof, mlaad_train_bonafide = count_labels(mlaad_path + 'train_meta.txt', 'MLAAD')
mlaad_dev_spooof, mlaad_dev_bonafide = count_labels(mlaad_path + 'dev_meta.txt', 'MLAAD')
mlaad_eval_spooof, mlaad_eval_bonafide = count_labels(mlaad_path + 'eval_meta.txt', 'MLAAD')
print(f"MLAAD Train: Spoof: {mlaad_train_spooof}, Bonafide: {mlaad_train_bonafide}")
print(f"MLAAD Dev: Spoof: {mlaad_dev_spooof}, Bonafide: {mlaad_dev_bonafide}")
print(f"MLAAD Eval: Spoof: {mlaad_eval_spooof}, Bonafide: {mlaad_eval_bonafide}")


dataset_dir = 'database/MLAAD_GB_v2'

if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
mlaand_v2_base_dir = r"database\mlaad_v2\MLAAD\fake"

#### READ META BY GENDER
meta_csv_paths = get_meta_csv_paths(mlaand_v2_base_dir)
speakers = read_meta(meta_csv_paths)
for gender, speaker in speakers.items():
    print(f"{gender}: {len(speaker)}")
    for s in speaker:
        print(s + ' ' + gender)