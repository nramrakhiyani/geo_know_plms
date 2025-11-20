import os
import sys
import codecs
from transformers import T5Tokenizer, T5ForConditionalGeneration

input_folder_path = sys.argv[1]
output_file_path = sys.argv[2]

print ('Loading T5 model')
t5_model_path = os.path.join('resources', 'transformers', 't5-base')
tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

output_file = codecs.open(output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')

file_list = os.listdir(input_folder_path)
for file_name in file_list:
	print ('Processing: ' + file_name)
	file_path = os.path.join(input_folder_path, file_name)
	file = codecs.open(file_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in file:
		line = line.strip()
		if(len(line) == 0):
			continue
		line_parts = line.split('\t')
		curr_sent = line_parts[0].replace('[MASK]', line_parts[1])

		input_ids = tokenizer('cola sentence: ' + curr_sent, return_tensors =  'pt').input_ids 
		output_sentence_ids = model.generate(input_ids)
		output_sentence = tokenizer.decode(output_sentence_ids[0], skip_special_tokens = True)

		if(output_sentence == 'acceptable'):
			output_file.write(file_name + '\tacceptable\t' + line + '\n')
		else:
			output_file.write(file_name + '\t' + output_sentence + '\t' + line + '\n')

output_file.close()