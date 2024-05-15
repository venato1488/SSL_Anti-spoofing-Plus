

# XLSDS Setup and Execution Guide

## 1. Set Up Environment

### Create and activate a conda environment:

```sh
$ conda create -n XLSDS python=3.7
$ conda activate XLSDS
```

### Install the necessary packages:

```sh
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Navigate to the fairseq directory (download from GitHub if not present):

```sh
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```

(Also available at: [fairseq GitHub](https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1))

```sh
$ pip install --editable ./
$ pip install -r requirements.txt
```

## 2. Model and Data Preparation

### Download the pre-trained XLS-R 300M model and place it in the root directory:

(Download link: [XLS-R 300M](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr))

### Download required datasets:

- [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336)
- [ASVspoof 2021 DF](https://zenodo.org/records/4835108#.YnDIb3YzZhE)
- [ASVspoof 2021 DF labels](https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz)
- [ITW](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa)
- [MLAAD v2](https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP?path=%2Fv2)
- [M-AILABS](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)

## 3. Data Organization

### Organize data according to the following structure:

```
database/
    |- DF
    |- ITW        
        |- wav 
        |- meta.csv 
    |- mlaad_v2   
        |- MLAAD
            |- fake
                |- ar
                |- bg
                |- cs
                |- ...
    |- MAILABS      
        |- de_DE
        |- en_UK
        |- en_US
        |- ...
```

### Extract 'release_in_the_wild.zip' into the ITW directory as indicated above.

### Compile MLAAD_GB by executing:

```sh
$ python dataset_prep.py
```

## 4. Training

### To train the model, run:

```sh
$ python main_XLSDS.py --lr=0.000001 --batch_size=14 --loss=WCE
```

## 5. Evaluation

### To evaluate the model on ASV, execute:

```sh
$ python main_XLSDS.py --is_eval_ASV --eval --model_path='PT_models/XLSDS.pth' --eval_output='eval_pre_trained_model_CM_scores_file_SSL_DF.txt'
```

### To evaluate on ITW, run:

```sh
$ python main_XLSDS.py --is_eval_ITW --eval --model_path='PT_models/XLSDS.pth' --eval_output='eval_pre_trained_model_CM_scores_file_SSL_DF.txt'
```

### To evaluate on MLAAD, run:

```sh
$ python main_XLSDS.py --is_eval_MLAAD_GB --eval --model_path='PT_models/XLSDS.pth' --eval_output='eval_pre_trained_model_CM_scores_file_SSL_DF.txt'
```

## 6. Compute EER

### To calculate the Equal Error Rate (EER) of the fine-tuned model, run:

```sh
$ python evaluate_DF.py score_obtained_after_evaluation.txt ./keys eval
```

Ensure each command is executed correctly and verify the paths for model and dataset locations as per your system configuration.
```
