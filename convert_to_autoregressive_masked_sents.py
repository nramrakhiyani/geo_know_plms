import os
import sys
import codecs

mask_files_folder_path = sys.argv[1]
output_folder_path = sys.argv[2]

total_sentences_converted = 0
mask_file_list = os.listdir(mask_files_folder_path)
for mask_file_name in mask_file_list:
	print ('Working on ' + mask_file_name)
	mask_file_path = os.path.join(mask_files_folder_path, mask_file_name)
	mask_file = codecs.open(mask_file_path, 'r', encoding = 'utf-8', errors = 'ignore')

	output_file_path = os.path.join(output_folder_path, mask_file_name)
	output_file = codecs.open(output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')

	num_mask_questions = 0
	for line in mask_file:
		line = line.strip()
		if(len(line.strip()) == 0):
			continue

		line_parts = line.split('\t')
		if(line_parts[0].endswith('[MASK].')):
			output_file.write(line_parts[0][:-1] + '\t' + line_parts[1] + '\n')
			num_mask_questions += 1
	print ('Sentence converted: ' + str(num_mask_questions))
	total_sentences_converted += num_mask_questions
	mask_file.close()
	output_file.close()
print ('Total Sentences Converted: ' + str(total_sentences_converted))