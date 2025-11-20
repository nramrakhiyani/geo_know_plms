import os
import sys

queries_output_folder = sys.argv[1]

list_subj_rels = [('Q12323', 'P17'), ('Q12323', 'P206'), ('Q165', 'P205'), ('Q4022', 'P205'), ('Q4421', 'P17'), ('Q46831', 'P17'), ('Q46831', 'P610'), ('Q515', 'P17'), ('Q6256', 'P30'), ('Q6256', 'P36'), ('Q8514', 'P17'), ('Q8514', 'P30'), ('Q150784', 'P17'), ('Q150784', 'P206'), ('Q194195', 'P17'), ('Q34038', 'P17'), ('Q179049', 'P17'), ('Q483110', 'P17'), ('Q1324633', 'P17'), ('Q1324633', 'P206'), ('Q695850', 'P17'), ('Q820477', 'P17'), ('Q1248784', 'P17'), ('Q159719', 'P17')]
list_lang = ['en', 'hi', 'mr', 'gu']

for lang in list_lang:
	curr_output_file_path = os.path.join(queries_output_folder, 'queries_' + lang + '.txt')
	curr_output_file = open(curr_output_file_path, 'w')
	for subj_rel in list_subj_rels:
		subj = subj_rel[0]
		rel = subj_rel[1]
		curr_output_file.write('Query\t' + subj + '\t' + rel + '\n')
		curr_output_file.write(
		'SELECT ?item ?itemLabel ?propId ?propVal ?propValLabel\n' + 
		'WHERE\n' +  
		'{\n' +
		'?item wdt:P31 wd:' + subj + '.\n' +
		'SERVICE wikibase:label { bd:serviceParam wikibase:language "' + lang + '".}\n' +
		'?item wikibase:sitelinks ?sitelinks.\n' +
		'?item ?propId ?propVal\n' +
		'VALUES ?propId {wdt:' + rel + '}\n' +
		'}  order by desc(?sitelinks)\n' +
		'LIMIT 200\n\n')
	curr_output_file.close()