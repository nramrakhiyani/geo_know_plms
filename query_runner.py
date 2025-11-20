import os
import sys
import codecs
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

queries_input_file_path = sys.argv[1]
queries_output_folder = sys.argv[2]

lang = queries_input_file_path[queries_input_file_path.rfind('_') + 1:queries_input_file_path.rfind('.')]
queries_input_file = codecs.open(queries_input_file_path, 'r', encoding = 'utf-8', errors = 'ignore')
curr_query = []
curr_subj = ''
curr_rel = ''

for line in queries_input_file:
	line = line.strip()
	if(len(line) == 0 and len(curr_query) > 0):
		print ('Working on query for: ' + curr_subj + '\t' + curr_rel)
		curr_query_str = '\n'.join(curr_query)
		sparql.setQuery(curr_query_str)
		sparql.setReturnFormat(JSON)
		results = sparql.query().convert()

		results_df = pd.io.json.json_normalize(results['results']['bindings'])
		output_path = os.path.join(queries_output_folder, lang, curr_subj + '_' + curr_rel + '.tsv')
		results_df[['itemLabel.value', 'propId.value', 'propValLabel.value']].to_csv(output_path, sep = "\t", index = False)
		curr_query = []
		curr_subj = ''
		curr_rel = ''

	elif(len(line) == 0):
		continue

	elif(line.startswith('Query')):
		line_parts = line.split('\t')
		curr_subj = line_parts[1]
		curr_rel = line_parts[2]

	else:
		curr_query.append(line)
