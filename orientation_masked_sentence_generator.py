import os 
import sys
import codecs

input_file_path = sys.argv[1]
output_folder_path = sys.argv[2]

direction_words = ['north', 'south', 'east', 'west', 'southwest', 'southeast', 'northeast', 'northwest']

output_file_path = os.path.join(output_folder_path, 'QFeature_POrient.txt')
output_file = codecs.open(output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')

input_file = codecs.open(input_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
found_trigger_with_dir_word = False
curr_sent = ''
curr_mask_locations = []
curr_wno = 0
for line in input_file:
	line = line.strip()
	if(len(line) == 0 and curr_sent != ''):
		curr_sent = curr_sent.strip()
		if(found_trigger_with_dir_word):
			found_trigger_with_dir_word = False
			for mask_loc in curr_mask_locations:
				curr_sent_parts = curr_sent.split()
				answer = curr_sent_parts[mask_loc]
				curr_sent_parts[mask_loc] = '[MASK]'
				output_file.write(curr_country + '\t' + ' '.join(curr_sent_parts) + '\t' + answer + '\n')
		curr_wno = 0
		curr_sent = ''
		curr_mask_locations = []
		continue
	elif(len(line) == 0):
		continue

	line_parts = line.split('\t')
	curr_country = line_parts[0]
	if(line_parts[1] in direction_words and line_parts[3] == 'trigger'):
		found_trigger_with_dir_word = True
		curr_mask_locations.append(curr_wno)

	curr_sent = curr_sent + ' ' + line_parts[1]
	curr_wno += 1

if(curr_sent != ''):
	if(found_trigger_with_dir_word):
		for mask_loc in curr_mask_locations:
			curr_sent_parts = curr_sent.split()
			answer = curr_sent_parts[mask_loc]
			curr_sent_parts[mask_loc] = '[MASK]'
			output_file.write(curr_country + '\t' + ' '.join(curr_sent_parts) + '\t' + answer + '\n')

input_file.close()
output_file.close()