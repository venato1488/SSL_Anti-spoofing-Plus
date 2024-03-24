import os
import csv
import shutil

def analyse_protocol(protocol_path):
    male = 0
    female = 0
    with open(protocol_path, 'r') as file:
        for line in file:
            speaker, file_name, label = line.split()
            if speaker.split('_')[-1] == 'male':
                male += 1 
            else:
                female += 1
    return male, female


def get_meta_csv_paths(base_dir):
    meta_csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "meta.csv":
                meta_csv_paths.append(os.path.join(root, file))
    return meta_csv_paths

def create_new_meta(meta_dir, speakers):
    meta_path = os.path.join(meta_dir, 'meta.txt')
    with open(meta_path , 'w') as file:
        for speaker, file_names in speakers.items():
            for file_name in file_names:
                # Assign the label based on the file number odd is spoof and even is bonafide
                label = 'bonafide' if int(file_name.split('_')[0]) % 2 == 0 else 'spoof'
                file.write(f"{speaker} {file_name} {label}\n")

            print(f"Speaker: {speaker} - Number of files: {len(file_names)}")
        file.close()  
    print(f"Total number of speakers: {len(speakers)}")


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
                
                gender = path_o.split('/')[2]
                #print(gender)
                speaker = path_o.split('/')[3]+'_male' if gender == 'male' else path_o.split('/')[3]+'_female'
                #print(speaker)
                # Extract the path to spoofed
                path_s =path[1:]

                filename_s = os.path.basename(path_s)
                filename_o = os.path.basename(path_o)

                # Append the counter to the filename
                new_filename_s = f"{file_counter}_{filename_s}"
                new_filename_o = f"{file_counter+1}_{filename_o}"
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(new_filename_s)
                speakers[speaker].append(new_filename_o)
                spoof_source_path = os.path.join('database/mlaad_v2/MLAAD'+path_s)
                orig_source_path = os.path.join('database/MAILABS/'+path_o)

                audio_file_dest_path = concat_dataset_dir + '/wav'
                if not os.path.exists(audio_file_dest_path):
                    os.mkdir(audio_file_dest_path)

                spoof_dest_path = os.path.join(audio_file_dest_path, new_filename_s)
                orig_dest_path = os.path.join(audio_file_dest_path, new_filename_o)

                # Copy and rename the files
             #   shutil.copy(spoof_source_path, spoof_dest_path)
             #   shutil.copy(orig_source_path, orig_dest_path)
                file_counter += 2
        
            meta_cvs_counter += 1
            #print(f"{meta_cvs_counter} meta.csv files processed out of {len(meta_csv_paths)}")
        """if meta_cvs_counter == 2:
            break"""
    create_new_meta(concat_dataset_dir, speakers)




dataset_dir = 'database/MLAAD_and_M-AILABS'
if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
mlaand_v2_base_dir = r"database\mlaad_v2\MLAAD\fake"

#meta_csv_paths = get_meta_csv_paths(mlaand_v2_base_dir)
#print(len(meta_csv_paths))
#copy_files_to_new_dir(meta_csv_paths, dataset_dir)

male, female = analyse_protocol(dataset_dir+'/meta.txt')
print(f"Male audio: {male}\nFemale audio: {female}")