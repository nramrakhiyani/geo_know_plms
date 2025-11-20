import re
import os
import sys
import torch
import codecs
import numpy as np
import unicodedata
from transformers import pipeline, AutoTokenizer

model_name = sys.argv[1]
mask_files_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]

#structured_output_file_path = sys.argv[5]
#structured_output_file = codecs.open(structured_output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')

model_path = os.path.join('resources', 'transformers', model_name)

#slight change in the mask token for roberta
mask_token = '[MASK]'
if(model_name.startswith('roberta') or model_name.startswith('bart') or model_name.startswith('xlnet')):
	mask_token = '<mask>'
elif(model_name.startswith('xlm')):
	mask_token = '<special1>'

tokenizer = AutoTokenizer.from_pretrained(model_path)
generator = pipeline('text-generation', model = model_path, tokenizer = tokenizer, torch_dtype = torch.bfloat16, trust_remote_code = True, device = 0, pad_token_id = tokenizer.eos_token_id)

overall_near1_hits = 0
overall_near5_hits = 0
overall_near10_hits = 0
total_num_questions = 0
mask_file_list = os.listdir(mask_files_folder_path)
for mask_file_name in mask_file_list:
	print ('Working on ' + mask_file_name)
	curr_near1_hits = 0
	curr_near5_hits = 0
	curr_near10_hits = 0
	curr_num_questions = 0

	mask_file_path = os.path.join(mask_files_folder_path, mask_file_name)
	mask_file = codecs.open(mask_file_path, 'r', encoding = 'utf-8', errors = 'ignore')

	output_file_path = os.path.join(output_folder_path, model_name, mask_file_name)
	output_file = codecs.open(output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')

	mask_questions = []
	for line in mask_file:
		line = line.strip()
		if(len(line.strip()) == 0):
			continue

		line_parts = line.split('\t')
		if(line_parts[0].endswith(mask_token + '.')):
			mask_questions.append((line_parts[0], line_parts[1]))
	mask_file.close()

	print ('Number of mask questions: ' + str(len(mask_questions)))

	for mqa in mask_questions:
		curr_num_questions += 1
		total_num_questions += 1
		mq = mqa[0]
		mq = mq.replace('[MASK]', mask_token)
		mq = unicodedata.normalize('NFKD', mq).encode('ascii', 'ignore').decode('utf-8')

		ma = mqa[1].split('~')
		ma_new = []
		for elem in ma:
			ma_new.append(unicodedata.normalize('NFKD', elem).encode('ascii', 'ignore').decode('utf-8'))
		ma = ma_new

		mq_input_for_gen = mq[0:mq.find(mask_token) - 1]

		#Adding instruction (Prompt Type 2)
		mq_input_for_gen = 'For the following sentence about geography, generate the most probable text to complete it. ' + mq_input_for_gen

		output = generator(mq_input_for_gen, do_sample = False, max_new_tokens = 30, return_full_text = False, temperature = 0.1)
		orig_gen_text = output[0]['generated_text'].strip()

		#Addition of cleaning of orig_gen_text
		gen_text = re.sub('\n', ' ', orig_gen_text)
		if(gen_text.find(' in ') > 0):
			temp_location_index = gen_text.find(' in ') + len(' in ')
			gen_text = gen_text[temp_location_index:]
		gen_text = re.sub('^[\.\,\!\?\_\' ]+|[\.\,\!\?\_\' ]+$', '', gen_text)

		output_file.write('===========================\n' + mq + '\t' + '~'.join(ma) + '\n')
		answer_found = False
		rank = 1
		top_10_answers = gen_text.split()
		for curr_answer in top_10_answers:
			output_file.write(str(rank) + '\t' + curr_answer + '\n')
			curr_answer_clean = curr_answer
			if(curr_answer.endswith('.') or curr_answer.endswith(',') or curr_answer.endswith('?') or curr_answer.endswith('!')):
				curr_answer_clean = curr_answer[:len(curr_answer) - 1]

			if(curr_answer in ma or curr_answer_clean in ma):
				answer_found = True
				break
			rank += 1

		if(answer_found):
			if(rank == 1):
				curr_near1_hits += 1
				overall_near1_hits += 1

			if(rank <= 5):
				curr_near5_hits += 1
				overall_near5_hits += 1

			if(rank <= 10):
				curr_near10_hits += 1
				overall_near10_hits += 1

	if(curr_num_questions > 0):
		output_file.write('===========================\n' + mask_file_name + '\n')
		output_file.write('Number of Questions: ' + str(curr_num_questions) + '\n')
		output_file.write('#Near 1 hits: ' + str(curr_near1_hits) + '\t' + str(float(curr_near1_hits/curr_num_questions)) + '\n')
		output_file.write('#Near 5 hits: ' + str(curr_near5_hits) + '\t' + str(float(curr_near5_hits/curr_num_questions)) + '\n')
		output_file.write('#Near 10 hits: ' + str(curr_near10_hits) + '\t' + str(float(curr_near10_hits/curr_num_questions)) + '\n')
		output_file.close()

		#structured_output_file.write(str(float(curr_near1_hits/curr_num_questions)) + '\n')
		#structured_output_file.write(str(float(curr_near5_hits/curr_num_questions)) + '\n')
		#structured_output_file.write(str(float(curr_near10_hits/curr_num_questions)) + '\n\n\n')
		print(str(float(curr_near1_hits/curr_num_questions)))
		print(str(float(curr_near5_hits/curr_num_questions)))
		print(str(float(curr_near10_hits/curr_num_questions)))

print ('Total Number of Questions: ' + str(total_num_questions) + '\n')
print ('#Near 1 hits: ' + str(overall_near1_hits) + '\t' + str(float(overall_near1_hits/total_num_questions)) + '\n')
print ('#Near 5 hits: ' + str(overall_near5_hits) + '\t' + str(float(overall_near5_hits/total_num_questions)) + '\n')
print ('#Near 10 hits: ' + str(overall_near10_hits) + '\t' + str(float(overall_near10_hits/total_num_questions)) + '\n')
output_file.close()