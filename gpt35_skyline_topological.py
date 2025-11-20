#Installation required #pip install openai==0.28
import re
import os
import sys
import openai
import codecs
import numpy as np
import unicodedata

mask_files_folder_path = sys.argv[1]
output_folder_path = sys.argv[2]

model_name = 'gpt35'
mask_token = '[MASK]'
openai.api_key = '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'

def alias_match(answer_list, curr_gen_text):
	if('USA' in answer_list and (curr_gen_text.find('United States of America') > 0 or curr_gen_text.find('United States') > 0 or curr_gen_text.find('the US of America') > 0)):
		return True
	elif('England' in answer_list and (curr_gen_text.find('United Kingdom') > 0 or curr_gen_text.find('Great Britain ') > 0 or curr_gen_text.find('the UK') > 0)):
		return True
	return False

def second_level_prompt(second_level_mq_input_for_gen, input_tokens, output_tokens, ma, output_file):
	response = openai.Completion.create(model = 'gpt-3.5-turbo-instruct', prompt = second_level_mq_input_for_gen, temperature = 0.1, max_tokens = 20)
	second_level_orig_gen_text = response['choices'][0]['text']
	#print (second_level_orig_gen_text)

	input_tokens.append(response['usage']['prompt_tokens'])
	output_tokens.append(response['usage']['completion_tokens'])

	#Addition of cleaning of second_level_orig_gen_text
	second_level_gen_text = re.sub('\n', ' ', second_level_orig_gen_text)
	if(second_level_gen_text.find(' in ') > 0):
		temp_location_index = second_level_gen_text.find(' in ') + len(' in ')
		second_level_gen_text = second_level_gen_text[temp_location_index:]
	second_level_gen_text = re.sub('^[\.\,\!\?\_\' ]+|[\.\,\!\?\_\' ]+$', '', second_level_gen_text)

	output_file.write('===========================\n')
	output_file.write('Q: ' + second_level_mq_input_for_gen + '\n')
	output_file.write('A: ' + second_level_gen_text + '\n')
	answer_found = False
	rank = 1
	top_10_answers = second_level_gen_text.split()
	for curr_answer in top_10_answers:
		output_file.write(str(rank) + '\t' + curr_answer + '\n')
		curr_answer_clean = curr_answer
		if(curr_answer.endswith('.') or curr_answer.endswith(',') or curr_answer.endswith('?') or curr_answer.endswith('!')):
			curr_answer_clean = curr_answer[:len(curr_answer) - 1]

		if(curr_answer in ma or curr_answer_clean in ma):
			answer_found = True
			break
		elif(alias_match(ma, second_level_gen_text)):
			answer_found = True
			break
		rank += 1
	return (answer_found, rank)

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

	input_tokens = []
	output_tokens = []

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
		if(line_parts[0].endswith(mask_token)):
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

		#output = generator(mq_input_for_gen, do_sample = False, max_new_tokens = 30, return_full_text = False, temperature = 0.1)
		response = openai.Completion.create(model = 'gpt-3.5-turbo-instruct', prompt = mq_input_for_gen, temperature = 0.1, max_tokens = 10)
		orig_gen_text = response['choices'][0]['text']

		input_tokens.append(response['usage']['prompt_tokens'])
		output_tokens.append(response['usage']['completion_tokens'])

		#Addition of cleaning of orig_gen_text
		gen_text = re.sub('\n', ' ', orig_gen_text)
		#if(gen_text.find(' in ') > 0):
		#	temp_location_index = gen_text.find(' in ') + len(' in ')
		#	gen_text = gen_text[temp_location_index:]
		gen_text = re.sub('^[\.\,\!\?\_\' ]+|[\.\,\!\?\_\' ]+$', '', gen_text)

		output_file.write('===========================\n' + mq + '\t' + '~'.join(ma) + '\n')
		output_file.write(gen_text + '\n')
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
		else:
			#print ('first level answer not found')
			if(mask_file_name.endswith('P17.txt')):
				second_level_mq_input_for_gen = 'In which country is ' + gen_text + ' located?'
				second_level_answer_found, second_level_rank = second_level_prompt(second_level_mq_input_for_gen, input_tokens, output_tokens, ma, output_file)

				if(second_level_answer_found):
					if(second_level_rank == 1):
						curr_near1_hits += 1
						overall_near1_hits += 1

					if(second_level_rank <= 5):
						curr_near5_hits += 1
						overall_near5_hits += 1

					if(second_level_rank <= 10):
						curr_near10_hits += 1
						overall_near10_hits += 1

	if(curr_num_questions > 0):
		output_file.write('===========================\n' + mask_file_name + '\n')
		output_file.write('Number of Questions: ' + str(curr_num_questions) + '\n')
		output_file.write('#Near 1 hits: ' + str(curr_near1_hits) + '\t' + str(float(curr_near1_hits/curr_num_questions)) + '\n')
		output_file.write('#Near 5 hits: ' + str(curr_near5_hits) + '\t' + str(float(curr_near5_hits/curr_num_questions)) + '\n')
		output_file.write('#Near 10 hits: ' + str(curr_near10_hits) + '\t' + str(float(curr_near10_hits/curr_num_questions)) + '\n')
		output_file.close()

		print(str(float(curr_near1_hits/curr_num_questions)))
		print(str(float(curr_near5_hits/curr_num_questions)))
		print(str(float(curr_near10_hits/curr_num_questions)))

	input_tokens_sum = sum(input_tokens)
	print (input_tokens_sum, '\t', input_tokens_sum * 0.0000015, '\t', input_tokens_sum * 0.0000015 * 83.38)
	output_tokens_sum = sum(output_tokens)
	print (output_tokens_sum, '\t', output_tokens_sum * 0.000002, '\t', output_tokens_sum * 0.000002 * 83.38)
	progress = input('Process next file?')
	if(progress == 'n'):
		break

print ('Total Number of Questions: ' + str(total_num_questions) + '\n')
print ('#Near 1 hits: ' + str(overall_near1_hits) + '\t' + str(float(overall_near1_hits/total_num_questions)) + '\n')
print ('#Near 5 hits: ' + str(overall_near5_hits) + '\t' + str(float(overall_near5_hits/total_num_questions)) + '\n')
print ('#Near 10 hits: ' + str(overall_near10_hits) + '\t' + str(float(overall_near10_hits/total_num_questions)) + '\n')
output_file.close()