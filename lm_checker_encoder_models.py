import os
import sys
import torch
import codecs
import numpy as np
import unicodedata
from transformers import AutoTokenizer,  AutoModelForMaskedLM, BartTokenizer, BartForConditionalGeneration, XLNetTokenizer, XLNetLMHeadModel

model_name = sys.argv[1]
uncased = sys.argv[2]
mask_files_folder_path = sys.argv[3]
output_folder_path = sys.argv[4]

model_path = os.path.join('resources', 'transformers', model_name)
if(model_name.startswith('bart')):
	tokenizer = BartTokenizer.from_pretrained(model_path)
	model = BartForConditionalGeneration.from_pretrained(model_path)
elif(model_name.lower().startswith('xlnet')):
	tokenizer = XLNetTokenizer.from_pretrained(model_path)
	model = XLNetLMHeadModel.from_pretrained(model_path)
else:
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForMaskedLM.from_pretrained(model_path)

#slight change in the mask token for roberta
mask_token = '[MASK]'
if(model_name.startswith('roberta') or model_name.startswith('bart') or model_name.startswith('xlnet')):
	mask_token = '<mask>'
elif(model_name.startswith('xlm')):
	mask_token = '<special1>'

overall_top5_hits = 0
overall_top10_hits = 0
overall_top50_hits = 0
overall_rr_val = 0.0
total_num_questions = 0
mask_file_list = os.listdir(mask_files_folder_path)
for mask_file_name in mask_file_list:
	print ('Working on ' + mask_file_name)
	curr_top5_hits = 0
	curr_top10_hits = 0
	curr_top50_hits = 0
	curr_rr_val = 0.0
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
		mask_questions.append((line_parts[0], line_parts[1]))
	mask_file.close()

	for mqa in mask_questions:
		curr_num_questions += 1
		total_num_questions += 1
		mq = mqa[0]
		if(uncased.lower() == 'true'):
			mq = mq.lower()
			mq = mq.replace('[mask]', mask_token)
			ma = mqa[1].lower().split('~')
		else:
			mq = mq.replace('[MASK]', mask_token)
			ma = mqa[1].split('~')

		mq = unicodedata.normalize('NFKD', mq).encode('ascii', 'ignore').decode('utf-8')
		ma_new = []
		for elem in ma:
			ma_new.append(unicodedata.normalize('NFKD', elem).encode('ascii', 'ignore').decode('utf-8'))
		ma = ma_new

		if(model_name.lower().startswith('xlnet')):
			input_ids = torch.tensor(tokenizer.encode(mq, add_special_tokens = False)).unsqueeze(0)

			perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype = torch.float)
			perm_mask[:, :, -1] = 1.0

			target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype = torch.float) 
			target_mapping[0, 0, -1] = 1.0

			outputs = model(input_ids, perm_mask = perm_mask, target_mapping = target_mapping)
			next_token_logits = outputs[0][0]
			softmaxed_output = next_token_logits.softmax(dim = -1)
			values, predictions = softmaxed_output.topk(50)

		else:
			inputs = tokenizer(mq, return_tensors = "pt")
			with torch.no_grad():
				logits = model(**inputs).logits

			# retrieve index of [MASK]
			mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
			#predicted_logits = logits[0, mask_token_index][0]
			probs = logits[0, mask_token_index].softmax(dim = -1)

			#partitioned_indices = np.argpartition(predicted_logits, -50)[-50:]
			#sorted_partitioned_indices = np.argsort(predicted_logits[partitioned_indices])
			#smaller_first_sorted_indices = partitioned_indices[sorted_partitioned_indices]
			#sorted_indices = torch.flip(smaller_first_sorted_indices, dims = [-1])
			values, predictions = probs.topk(50)

		output_file.write('===========================\n' + mq + '\t' + '~'.join(ma) + '\n')
		rank = 1
		answer_found = False
		top_50_answers = tokenizer.decode(predictions[0]).split()
		for curr_answer in top_50_answers:
			output_file.write(str(rank) + '\t' + curr_answer + '\n')
			if(curr_answer in ma):
				answer_found = True
				break
			rank += 1

		if(answer_found):
			if(rank <= 5):
				curr_top5_hits += 1
				overall_top5_hits += 1

			if(rank <= 10):
				curr_top10_hits += 1
				overall_top10_hits += 1

			curr_top50_hits += 1
			overall_top50_hits += 1

			curr_rr_val += float(1/rank)
			overall_rr_val += float(1/rank)

	mrr = curr_rr_val/float(curr_num_questions) 
	output_file.write('===========================\n' + mask_file_name + '\n')
	output_file.write('Number of Questions: ' + str(curr_num_questions) + '\n')
	output_file.write('#Top 5 hits: ' + str(curr_top5_hits) + '\t' + str(float(curr_top5_hits/curr_num_questions)) + '\n')
	output_file.write('#Top 10 hits: ' + str(curr_top10_hits) + '\t' + str(float(curr_top10_hits/curr_num_questions)) + '\n')
	output_file.write('#Top 50 hits: ' + str(curr_top50_hits) + '\t' + str(float(curr_top50_hits/curr_num_questions)) + '\n')
	output_file.write('Mean Reciprocal rank: ' + str(mrr))
	output_file.close()
	print(str(float(curr_top5_hits/curr_num_questions)))
	print(str(float(curr_top10_hits/curr_num_questions)))
	print(str(float(curr_top50_hits/curr_num_questions)))
	print(str(mrr))

overall_mrr = overall_rr_val/float(total_num_questions) 
print ('Total Number of Questions: ' + str(total_num_questions) + '\n')
print ('#Top 5 hits: ' + str(overall_top5_hits) + '\t' + str(float(overall_top5_hits/total_num_questions)) + '\n')
print ('#Top 10 hits: ' + str(overall_top10_hits) + '\t' + str(float(overall_top10_hits/total_num_questions)) + '\n')
print ('#Top 50 hits: ' + str(overall_top50_hits) + '\t' + str(float(overall_top50_hits/total_num_questions)) + '\n')
print ('Overall Reciprocal rank: ' + str(overall_mrr))