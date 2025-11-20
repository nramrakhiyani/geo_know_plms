import os
import re
import sys
import math
import codecs
import random
import unicodedata
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import cuda
from torch import tensor
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ["TOKENIZER_PARALLELISM"] = "false"

def read_instances(input_file_path):
    curr_instances = []
    input_file = codecs.open(input_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
    for line in input_file:
        line = line.strip()
        if(len(line) == 0):
            continue
    
        line_parts = line.split('\t')
        if(len(line_parts) < 2):
            print ('Skipping: ' + line)
            continue

        if(line_parts[1].find('~') > 0):
            multiple_targets = line_parts[1].split('~')
            for curr_target in multiple_targets:
                if(not curr_target.find(' ') > 0):
                    curr_instances.append((line_parts[0], curr_target))
        else:
            curr_instances.append((line_parts[0], line_parts[1]))
    input_file.close()
    return curr_instances

def read_data(folder_path, oversample_orientation_data):
    all_instances = []
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        curr_file_path = os.path.join(folder_path, file_name)
        curr_instances = read_instances(curr_file_path)
        all_instances.extend(curr_instances)
    
        #Adding data again
        if(oversample_orientation_data > 0 and not (file_name.startswith('Q') and file_name != 'Qatar.txt')):
            for k in range(oversample_orientation_data - 1):
                all_instances.extend(curr_instances)
    return all_instances

def control_trainable_parameters(model):
    total_trainable_parameters = 0

    for name, param in model.named_parameters():
        if(name.startswith('bert.encoder.layer.23')):
            param.required_grad = True
        else:
            param.requires_grad = False
    
    for name, param in model.named_parameters():
        if(param.requires_grad):
            print ('Trainable: ', name)
            total_trainable_parameters += param.numel()
        else:
            print ('Non-trainable: ', name)
    print ('Total trainable parameters: ', total_trainable_parameters)

class GeoMLMDataset(Dataset):
    def __init__(self, instances, tokenizer, max_len):
        self.num_total_instances = 0
        self.num_skipped_instances = 0
        self.dataset_item_dict = {}
    
        for k in range(len(instances)):
            curr_instance = instances[k]
            masked_text = curr_instance[0]
            mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
            tokenized_masked_text = tokenizer(masked_text, return_tensors = 'pt', padding = 'max_length', max_length = max_len, truncation = True)

            curr_label = tokenized_masked_text['input_ids'][0].clone()
            curr_label[:] = -100
            grounding_token_ids = tokenizer.convert_tokens_to_ids(curr_instance[1])
            if((not isinstance(grounding_token_ids, list)) and grounding_token_ids != -100):
                mask_index = (tokenized_masked_text['input_ids'][0] == mask_id).nonzero(as_tuple = True)[0][0].item()
                curr_label[mask_index] = grounding_token_ids
                self.dataset_item_dict[self.num_total_instances] = {
                    'sentence':masked_text + '\t' + curr_instance[1],
                    'input_ids':tokenized_masked_text['input_ids'][0],
                    'attention_mask':tokenized_masked_text['attention_mask'][0],
                    'token_type_ids':tokenized_masked_text['token_type_ids'][0],
                    'labels':curr_label
                }
            else:
                self.num_skipped_instances += 1
                print ('Skipping this training instance because of multi-token breaking of ' + curr_instance[1] + '\t' + str(grounding_token_ids))

            self.num_total_instances += 1
    
    def __len__(self):
        return self.num_total_instances
    
    def __getitem__(self, item):
        return self.dataset_item_dict[item]

base_dir = ''
repo_dir = ''
orig_models_path = 'models'

base_model_name = sys.argv[1]
ft_model_name = sys.argv[2]
train_dataset_path = sys.argv[3]
valid_dataset_path = sys.argv[4]
oversample_orientation_data_str = sys.argv[5]
oversample_orientation_data = int(oversample_orientation_data_str)

base_model_path = os.path.join(base_dir, orig_models_path, base_model_name)
ft_model_save_path = os.path.join(base_dir, repo_dir, 'models', ft_model_name)

train_folder_path = os.path.join(base_dir, repo_dir, train_dataset_path)
valid_folder_path = os.path.join(base_dir, repo_dir, valid_dataset_path)

device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')

#Initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

#Initializing model
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_path)
mlm_model.to(device)
control_trainable_parameters(mlm_model)

#Reading data
print ('Reading train data')
training_instances = read_data(train_folder_path, oversample_orientation_data)
train_dataset = GeoMLMDataset(training_instances, tokenizer, max_len = 64)
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 4)

#Training loop
print ('Training')
lr = 2e-5 #3e-4
num_epochs = 25
optimizer = AdamW(params = mlm_model.parameters(), lr = lr)

best_top5 = 0.0
mask_file_to_results = {}
for epoch in range(num_epochs):
    print ('****************** Running epoch ' + str(epoch + 1))

    mlm_model.train()
    for data in tqdm(train_dataloader, total = len(train_dataloader), position = 0, leave = True):
        for k, v in data.items():
            if(k != 'sentence'):
                data[k] = v.to(device)
        
        optimizer.zero_grad()
        outputs = mlm_model(input_ids = data['input_ids'], attention_mask = data['attention_mask'], token_type_ids = data['token_type_ids'], labels = data['labels'])
        curr_loss = outputs.loss
        curr_loss.backward()
        optimizer.step()
    
    print ('Evaluating')
    mlm_model.eval()
    orient_top3_hits = 0
    topo_top5_hits = 0
    total_num_questions_topological = 0
    total_num_questions_orientation = 0
    mask_file_list = os.listdir(valid_folder_path)

    mask_file_to_results = {}
    for mask_file_name in mask_file_list:
        mask_file_to_results[mask_file_name] = {'top3': 0, 'top5': 0, 'o_total': 0, 't_total': 0}
        mask_file_path = os.path.join(valid_folder_path, mask_file_name)
        
        mask_questions = []
        mask_file = codecs.open(mask_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
        for line in mask_file:
            line = line.strip()
            if(len(line) == 0):
                continue

            line_parts = line.split('\t')
            mask_questions.append((line_parts[0], line_parts[1]))
        mask_file.close()

        orientation_file = True
        if(mask_file_name.startswith('Q') and mask_file_name != 'Qatar.txt'):
            orientation_file = False
            total_num_questions_topological += len(mask_questions)
            mask_file_to_results[mask_file_name]['t_total'] = len(mask_questions)
        else:
            total_num_questions_orientation += len(mask_questions)
            mask_file_to_results[mask_file_name]['o_total'] = len(mask_questions)
        
        for mqa in mask_questions:
            mq = mqa[0]
            ma = mqa[1].split('~')
            
            mq = unicodedata.normalize('NFKD', mq).encode('ascii', 'ignore').decode('utf-8')
            ma_new = []
            for elem in ma:
                ma_new.append(unicodedata.normalize('NFKD', elem).encode('ascii', 'ignore').decode('utf-8'))
            ma = ma_new

            inputs = tokenizer(mq, return_tensors = "pt")
            inputs.to(device)
            with torch.no_grad():
                logits = mlm_model(**inputs).logits

            # retrieve index of [MASK]
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple = True)[0]
            probs = logits[0, mask_token_index].softmax(dim = -1)

            try:
                values, predictions = probs.topk(10)

                rank = 1
                answer_found = False
                top_10_answers = tokenizer.decode(predictions[0]).split()
                for curr_answer in top_10_answers:
                    if(curr_answer in ma):
                        answer_found = True
                        break
                    rank += 1

                if(answer_found):
                    if(rank <= 5 and orientation_file is False):
                        topo_top5_hits += 1
                        mask_file_to_results[mask_file_name]['top5'] += 1
                    elif(rank <= 3 and orientation_file is True):
                        orient_top3_hits += 1
                        mask_file_to_results[mask_file_name]['top3'] += 1
            except:
                print ('Skipping mq: ' + mq)

    for mask_file_name in mask_file_to_results:
        print (mask_file_name, '~', mask_file_to_results[mask_file_name]['top3'], '~', mask_file_to_results[mask_file_name]['o_total'], '~', mask_file_to_results[mask_file_name]['top5'], '~', mask_file_to_results[mask_file_name]['t_total'])
    
    if(total_num_questions_orientation == 0):
        curr_top3 = 0.0
    else:
        curr_top3 = float(orient_top3_hits) / float(total_num_questions_orientation)
    
    curr_top5 = float(topo_top5_hits) / float(total_num_questions_topological)
    if(best_top5 < curr_top5):
        best_top5 = curr_top5
        print ('Saving the model')
        mlm_model.save_pretrained(ft_model_save_path)
    
    print ('Epoch ' + str(epoch + 1) + '\tOrientation: ' + str(orient_top3_hits) + '\t' + str(total_num_questions_orientation))
    print ('Epoch ' + str(epoch + 1) + '\tTopological: ' + str(topo_top5_hits) + '\t' + str(total_num_questions_topological))
    print ('Epoch ' + str(epoch + 1) + '\t' + str(curr_top5) + '\t' + str(curr_top3))