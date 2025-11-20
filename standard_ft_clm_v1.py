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

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["TOKENIZER_PARALLELISM"] = "false"

def control_trainable_parameters(model):
    total_trainable_parameters = 0

    for name, param in model.named_parameters():
        if(name.startswith('transformer.h.31')):
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

        if(line_parts[0].endswith('[MASK].')):
            curr_text = line_parts[0].replace('[MASK].', '')
            if(line_parts[1].find('~') > 0):
                multiple_targets = line_parts[1].split('~')
                for curr_target in multiple_targets:
                    curr_instances.append((curr_text, curr_target))
            else:
                curr_instances.append((curr_text, line_parts[1]))
    input_file.close()
    return (curr_instances)

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

class GeoCLMDataset(Dataset):
    def __init__(self, instances, tokenizer, max_len):
        self.num_total_instances = 0
        self.num_skipped_instances = 0
        self.dataset_item_dict = {}
    
        instruction = 'For the following sentence about geography, generate the most probable text to complete it. '

        for k in range(len(instances)):
            curr_instance = instances[k]
            complete_text = instruction + curr_instance[0] + ' ' + curr_instance[1] + '.'

            #masked_text = curr_instance[0]
            #mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
            #tokenized_masked_text = tokenizer(masked_text, return_tensors = 'pt', padding = 'max_length', max_length = max_len, truncation = True)
            tokenized_complete_text = tokenizer(complete_text, return_tensors = 'pt', padding = 'max_length', max_length = max_len, truncation = True)

            curr_label = tokenized_complete_text['input_ids'][0].clone()
            self.dataset_item_dict[self.num_total_instances] = {
                    'sentence':complete_text,
                    'input_ids':tokenized_complete_text['input_ids'][0],
                    'attention_mask':tokenized_complete_text['attention_mask'][0],
                    'labels':curr_label
            }
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

#Setting padding side
if(any(k in base_model_path for k in ['gpt', 'opt', 'bloom'])):
    padding_side = 'left'
else:
    padding_side = 'right'

#Initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side = padding_side)
tokenizer.pad_token = tokenizer.eos_token

#Initializing model
clm_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code = True)
clm_model.to(device)
control_trainable_parameters(clm_model)
clm_model.config.pad_token_id = tokenizer.eos_token_id

#Reading data
print ('Reading train data')
training_instances = read_data(train_folder_path, oversample_orientation_data)
train_dataset = GeoCLMDataset(training_instances, tokenizer, max_len = 128)
train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 4)

#Read validation data
validation_instances = read_data(valid_folder_path, 0)

#Training loop
print ('Training')
lr = 2e-5 #3e-4
num_epochs = 10
optimizer = AdamW(params = clm_model.parameters(), lr = lr)

best_top5 = 0.0
for epoch in range(num_epochs):
    print ('****************** Running epoch ' + str(epoch + 1))

    clm_model.train()
    for data in tqdm(train_dataloader, total = len(train_dataloader), position = 0, leave = True):
        for k, v in data.items():
            if(k != 'sentence'):
                data[k] = v.to(device)
        
        optimizer.zero_grad()
        outputs = clm_model(input_ids = data['input_ids'], attention_mask = data['attention_mask'], token_type_ids = data['token_type_ids'], labels = data['labels'])
        curr_loss = outputs.loss
        curr_loss.backward()
        optimizer.step()
    
    print ('Evaluating')
    clm_model.eval()
    orient_top3_hits = 0
    topo_top5_hits = 0
    total_num_questions_topological = 0
    total_num_questions_orientation = 0
    for validation_instance in validation_instances:
        mq = validation_instance[0]
        mq = unicodedata.normalize('NFKD', mq).encode('ascii', 'ignore').decode('utf-8')
            
        ma = validation_instance[1].lower().split('~')
        ma_new = []
        for elem in ma:
            ma_new.append(unicodedata.normalize('NFKD', elem).encode('ascii', 'ignore').decode('utf-8'))
        ma = ma_new

        orientation_file = False
        if(any(ma_elem in ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'] for ma_elem in ma_new)):
            orientation_file = True
            total_num_questions_orientation += 1
        else:
            total_num_questions_topological += 1

        #Prompt type 1
        #mq_input_for_gen = 'Complete the following sentence: ' + mq
    
        #Prompt type 2
        mq_input_for_gen = 'For the following sentence about geography, generate the most probable text to complete it. ' + mq

        #Prompt type 3
        #mq_input_for_gen = 'Generate the most probable text to complete the following sentence. ' + mq

        #Prompt type 4
        #mq_input_for_gen = 'Complete the following geography fact. ' + mq

        #Prompt type 5
        #mq_input_for_gen = 'Answer the question: ' + mq + '?'
        
        generator = pipeline('text-generation', model = clm_model, tokenizer = tokenizer, torch_dtype = torch.bfloat16, trust_remote_code = True, device = 0, pad_token_id = tokenizer.eos_token_id)

        output = generator(mq_input_for_gen, do_sample = False, max_new_tokens = 30, return_full_text = False, temperature = 0.1)
        orig_gen_text = output[0]['generated_text'].strip()

        #Addition of cleaning of orig_gen_text
        gen_text = re.sub('\n', ' ', orig_gen_text)
        if(gen_text.find(' in ') > 0):
            temp_location_index = gen_text.find(' in ') + len(' in ')
            gen_text = gen_text[temp_location_index:]
        gen_text = re.sub(r'^[\.\,\!\?\_\' ]+|[\.\,\!\?\_\' ]+$', '', gen_text)

        rank = 1
        answer_found = False
        top_10_answers = gen_text.split()
        for curr_answer in top_10_answers:
            curr_answer_clean = curr_answer[:]
            if(curr_answer.endswith('.') or curr_answer.endswith(',') or curr_answer.endswith('?') or curr_answer.endswith('!')):
                curr_answer_clean = curr_answer[:len(curr_answer) - 1]
                    
            if(curr_answer in ma or curr_answer_clean in ma):
                answer_found = True
                break
            rank += 1

        if(answer_found):
            if(rank <= 5 and orientation_file is False):
                topo_top5_hits += 1
            elif(rank <= 3 and orientation_file is True):
                orient_top3_hits += 1
    
    if(total_num_questions_orientation == 0):
        curr_top3 = 0.0
    else:
        curr_top3 = float(orient_top3_hits) / float(total_num_questions_orientation)
    
    curr_top5 = float(topo_top5_hits) / float(total_num_questions_topological)
    if(best_top5 < curr_top5):
        best_top5 = curr_top5
        print ('Saving the model')
        clm_model.save_pretrained(ft_model_save_path)
    
    print ('Epoch ' + str(epoch + 1) + '\tOrientation: ' + str(orient_top3_hits) + '\t' + str(total_num_questions_orientation))
    print ('Epoch ' + str(epoch + 1) + '\tTopological: ' + str(topo_top5_hits) + '\t' + str(total_num_questions_topological))
    print ('Epoch ' + str(epoch + 1) + '\t' + str(curr_top5) + '\t' + str(curr_top3))