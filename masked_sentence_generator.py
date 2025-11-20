import os
import re
import sys
import codecs

entity_type_to_properties_file_path = sys.argv[1]
wikidata_query_output_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]
resources_path = sys.argv[4]

wikidata_stub_regex = re.compile(r'Q\d+')
num_post_subject_entity_replacements = 0

#Read resources
mappings = {}
mappings_file_path = os.path.join(resources_path, 'mappings', 'entity_mappings_en.txt')
mappings_file = codecs.open(mappings_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
header = True
for line in mappings_file:
	if(header):
		header = False
		continue

	line = line.strip()
	if(len(line) == 0):
		continue

	line_parts = line.split('\t')
	key_to_val = {}
	for part in line_parts[1:]:
		part_split = part.split('~')
		key_to_val[part_split[0]] = part_split[1]
	mappings[line_parts[0]] = key_to_val
mappings_file.close()

def check_if_wikidata_stub(input_val):
	if(wikidata_stub_regex.match(input_val)):
		return True
	return False

def generate_multi_entity_masked_sentence_old():
	curr_masked_sents = []

	#Adding mask to b_vals
	first_b_val = b_vals[0]
	if(len(first_b_val.split()) > 0):
		if(first_b_val in mappings):
			first_b_val = mappings[first_b_val]

	second_b_val = b_vals[1]
	if(len(second_b_val.split()) > 0):
		if(second_b_val in mappings):
			second_b_val = mappings[second_b_val]

	curr_masked_sent = template.replace('$A$', a_val)
	curr_masked_sent = curr_masked_sent.replace('$B_MULTI$', b_vals[0] + ' and ' + b_vals[1])

	#Adding mask to b_vals
	"""first_b_val = b_vals[0]
	if(len(first_b_val.split()) == 1):
		rest_b_vals = b_vals.remove()
		curr_masked_sent = template.replace('$A$', a_val)
		curr_masked_sent = curr_masked_sent.replace('$B_MULTI$', '[MASK] and ' + b_vals[1])
		curr_masked_sents.append((curr_masked_sent, first_b_val))

	second_b_val = b_vals[1]
	if(len(second_b_val.split()) == 1):
		curr_masked_sent = template.replace('$A$', a_val)
		curr_masked_sent = curr_masked_sent.replace('$B_MULTI$', b_vals[0] + ' and [MASK]')
		curr_masked_sents.append((curr_masked_sent, second_b_val))"""

def apply_mapping_replacements(input_list, mapping_type):
	output_list = []
	for elem in input_list:
		if(elem in mappings[mapping_type]):
			output_list.append(mappings[mapping_type][elem])
		else:
			output_list.append(elem)
	output_list = list(set(output_list))
	return output_list

def apply_specific_template_checks_old(template, input_val):
	if((template.find('forest') > 0 and input_val.lower().endswith(' forest')) or (template.find('river') > 0 and input_val.lower().endswith(' river')) or (template.find('desert') > 0 and input_val.lower().endswith(' desert')) or (template.find('dam') > 0 and input_val.lower().endswith(' dam')) or (template.find('mountain') > 0 and input_val.lower().endswith(' mountains'))):
		input_val_parts = input_val.split()
		input_val = ' '.join(input_val_parts[:-1])
	return input_val

def check_sublist(larger_list, smaller_list):
	sublist = False
	smaller_list_len = len(smaller_list)
	smaller_list_str = '_'.join(smaller_list)
	for k in range(len(larger_list) - smaller_list_len):
		larger_list_focus_str = '_'.join(larger_list[k:k+smaller_list_len])
		if(smaller_list_str == larger_list_focus_str):
			sublist = True
			break
	return sublist

def apply_specific_template_checks(template, input_val, type):
	new_template = template
	if(type == 'subject'):
		template_parts = template.split()
		for specific_token in token_to_variations:
			specific_token_len = len(specific_token.split())
			variations = token_to_variations[specific_token]
			if((specific_token_len == 1 and specific_token in template_parts) or (specific_token_len > 1 and check_sublist(template_parts, specific_token.split()))):
				input_val_parts = set(input_val.lower().split())
				if(len(input_val_parts.intersection(variations)) > 0):
					new_template = template.replace(' ' + specific_token, '', 1)

	else:
		input_val_parts = input_val.lower().split()
		set_input_val_parts = set(input_val_parts)
		
		#currently only for 'river' in case of the object entity
		set_river = set(['river', 'gulf', 'sea', 'canal', 'reservoir', 'creek', 'lake'])
		if(len(set_river.intersection(set_input_val_parts)) > 0):
			new_template = template.replace(' river', '', 1)
	return new_template

#specific pattern based correction after T5 correction exercise
def apply_specific_template_checks_post_subject_entity_replacement(template):
	global num_post_subject_entity_replacements
	new_template = re.sub(r'(Forests|forests|mountains|Mountains|Hills|hills|Mines|mines|Sanctuaries|sanctuaries|islands|Islands) is located', r'\1 are located', template)
	new_template = re.sub(r'(mountains|Mountains|Hills|hills) runs through', r'\1 run through', new_template)
	if(new_template != template):
		num_post_subject_entity_replacements += 1
	return(new_template)

def get_mask_position_in_obj_entity(input_val, property):
	mask_position = -1
	input_val_parts = input_val.lower().split()
	if(property in ['P17', 'P30', 'P205']):
		if(len(input_val_parts) == 1):
			mask_position = 0
	elif(property in ['P36']):
		if(input_val_parts[0] in ['port', 'sao', 'santo', 'st.', 'new']):
			mask_position = 1
		else:
			mask_position = 0
	elif(property in ['P610']):
		if(input_val_parts[0] in ['mont', 'mount']):
			mask_position = 1
		else:
			mask_position = 0
	else:
		if(input_val.lower().find(' of ') > 0):
			temp_mask_position = input_val_parts.index('of') + 1
			if(input_val_parts[temp_mask_position] in ['the', 'a', 'an']):
				mask_position = temp_mask_position + 1
			else:
				mask_position = temp_mask_position
		else:
			mask_position = 0
	return mask_position

def check_all_bvals_multiword(list_vals):
	all_bvals_multiword = True
	for val in list_vals:
		val_parts = val.split()
		if(len(val_parts) == 1):
			all_bvals_multiword = False
			break
	return all_bvals_multiword

def generate_multi_entity_masked_sentence(template, a_val, b_vals, mapping_type):
	curr_masked_sents = []

	if(check_all_bvals_multiword(b_vals)):
		return curr_masked_sents

	template = apply_specific_template_checks(template, a_val, 'subject')

	for k in range(len(b_vals)):
		curr_b_val = b_vals[k]
		curr_b_val_parts = curr_b_val.split()
		if(len(curr_b_val_parts) == 1):
			rest_b_vals = list(b_vals)
			rest_b_vals.remove(curr_b_val)

			curr_b_val = apply_mapping_replacements([curr_b_val], mapping_type)[0]
			rest_b_vals = apply_mapping_replacements(rest_b_vals, mapping_type)

			curr_masked_sent = template.replace('$A$', a_val)
			curr_masked_sent = apply_specific_template_checks_post_subject_entity_replacement(curr_masked_sent)
			curr_masked_sent = curr_masked_sent.replace('$B_MULTI$', curr_b_val + ' and [MASK]')
			curr_masked_sents.append((curr_masked_sent, rest_b_vals))

	return curr_masked_sents

def generate_single_entity_masked_sentence(template, a_val, b_vals, mapping_type, property):
	curr_masked_sents = []

	curr_b_val = b_vals[0]
	template = apply_specific_template_checks(template, curr_b_val, 'object')

	template = apply_specific_template_checks(template, a_val, 'subject')
	curr_masked_sent = template.replace('$A$', a_val)
	curr_masked_sent = apply_specific_template_checks_post_subject_entity_replacement(curr_masked_sent)

	curr_b_val = apply_mapping_replacements([curr_b_val], mapping_type)[0]
	if(len(curr_b_val.split()) == 1):
		curr_masked_sent = curr_masked_sent.replace('$B$', '[MASK]')
		curr_masked_sents.append((curr_masked_sent, curr_b_val))
	else:
		mask_position = get_mask_position_in_obj_entity(curr_b_val, property)
		if(mask_position != -1):
			curr_b_val_parts = curr_b_val.split()
			masked_b_part = curr_b_val_parts[mask_position]
			curr_b_val_parts[mask_position] = '[MASK]'
			new_b_val = ' '.join(curr_b_val_parts)
			curr_masked_sent = curr_masked_sent.replace('$B$', new_b_val)
			curr_masked_sents.append((curr_masked_sent, masked_b_part))

	return curr_masked_sents

#Reading entity_type_to_properties_file
entity_type_to_properties_file = codecs.open(entity_type_to_properties_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
entity_type_property_to_templates = {}

#specific removals
entity_to_specific_remove = {}
specific_removal_file_path = os.path.join(resources_path, 'misc', 'removals.txt')
specific_removal_file = codecs.open(specific_removal_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
for line in specific_removal_file:
	line = line.strip()
	if(len(line) == 0):
		continue

	line_parts = line.split('\t')
	entity_to_specific_remove[line_parts[0]] = line_parts[1:]
specific_removal_file.close()

#specific muting of tokens in templates
token_to_variations = {}
token_to_variations_file_path = os.path.join(resources_path, 'misc', 'template_token_muting.txt')
token_to_variations_file = codecs.open(token_to_variations_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
for line in token_to_variations_file:
	line = line.strip()
	if(len(line) == 0):
		continue

	line_parts = line.split('\t')
	token_to_variations[line_parts[0]] = set(line_parts[1].split(', '))
token_to_variations_file.close()

header = True
for line in entity_type_to_properties_file:
	if(header):
		header = False
		continue

	line = line.strip()
	if(len(line) == 0):
		continue

	line_parts = line.split('\t')
	curr_entity_type = line_parts[1]
	curr_property = line_parts[2]
	curr_property = curr_property[:curr_property.find('~')]
	template_type = line_parts[3]
	template1 = line_parts[4]
	template2 = line_parts[5]
	if(template1 == 'NA' and template2 == 'NA'):
		continue

	curr_templates = {}
	if(curr_entity_type + '_' + curr_property in entity_type_property_to_templates):
		curr_templates = entity_type_property_to_templates[curr_entity_type + '_' + curr_property]

	if(template1 != 'NA' and template2 == 'NA'):
		curr_templates[template_type] = [template1]
	else:
		curr_templates[template_type] = [template1, template2]

	entity_type_property_to_templates[curr_entity_type + '_' + curr_property] = curr_templates
entity_type_to_properties_file.close()

#Preparing masked sentences
total_masked_sentences = 0
for entity_type_property in entity_type_property_to_templates:
	wikidata_query_output_file_path = os.path.join(wikidata_query_output_folder_path, entity_type_property + '.tsv')
	wikidata_query_output_file = codecs.open(wikidata_query_output_file_path, 'r', encoding = 'utf-8', errors = 'ignore')

	entity = entity_type_property.split('_')[0]
	property = entity_type_property.split('_')[1]
	if(property == 'P30'): #continent
		mapping_type = 'continent'
	else:
		mapping_type = 'country'

	a_val_to_b_vals = {}

	header = True
	seen_lines = {}
	for line in wikidata_query_output_file:
		if(header):
			header = False
			continue

		line = line.strip()
		if(len(line) == 0 or line in seen_lines):
			continue

		seen_lines[line] = ''

		line_parts = line.split('\t')
		a_val = line_parts[0]
		if(entity in entity_to_specific_remove and a_val in entity_to_specific_remove[entity]):
			continue

		a_val = re.sub(' \(.*?\)', '', a_val)
		b_val = line_parts[2]
		b_val = re.sub(' \(.*?\)', '', b_val)

		if(check_if_wikidata_stub(a_val) or check_if_wikidata_stub(b_val)):
			continue

		if(a_val not in a_val_to_b_vals):
			a_val_to_b_vals[a_val] = []
		a_val_to_b_vals[a_val].append(b_val)

	wikidata_query_output_file.close()

	num_masked_sents = 0
	output_file_path = os.path.join(output_folder_path, entity_type_property + '.txt')
	output_file = codecs.open(output_file_path, 'w', encoding = 'utf-8', errors = 'ignore')
	curr_templates = entity_type_property_to_templates[entity_type_property]
	for a_val in a_val_to_b_vals:
		b_vals = a_val_to_b_vals[a_val]
		if(len(b_vals) > 1 and 'MULTI' in curr_templates):
			templates = curr_templates['MULTI']
			for template in templates:
				curr_masked_sents = generate_multi_entity_masked_sentence(template, a_val, b_vals, mapping_type)
				for curr_masked_sent in curr_masked_sents:
					output_file.write(curr_masked_sent[0] + '\t' + '~'.join(curr_masked_sent[1]) + '\n')
					num_masked_sents += 1

		elif(len(b_vals) == 1 and 'SINGLE' in curr_templates):
			templates = curr_templates['SINGLE']
			for template in templates:
				curr_masked_sents = generate_single_entity_masked_sentence(template, a_val, b_vals, mapping_type, property)
				for curr_masked_sent in curr_masked_sents:
					output_file.write(curr_masked_sent[0] + '\t' + curr_masked_sent[1] + '\n')
					num_masked_sents += 1
	output_file.close()
	print ('Number of masked Sents in ' + entity_type_property + ': ' + str(num_masked_sents))
	total_masked_sentences += num_masked_sents
print ('Number of total masked Sents: ' + str(total_masked_sentences))
print ('Number of corrective replacements: ' + str(num_post_subject_entity_replacements))