# GeoKnow_PLMs
The repo consists of multiple parts of the implementation and contributions in the papers:
1. Nitin Ramrakhiyani, Vasudeva Varma, Girish Keshav Palshikar, and Sachin Pawar. **Gauging, enriching and applying geography knowledge in Pre-trained Language Models.** Information Processing & Management 62, no. 1 (2025): 103892.
2. Nitin Ramrakhiyani, Vasudeva Varma, Girish Palshikar, and Sachin Pawar. **Zero-shot Probing of Pretrained Language Models for Geography Knowledge.** In Proceedings of the 4th Workshop on Evaluation and Comparison of NLP Systems, pp. 49-61. 2023

## Introduction
There are multiple aspects of the work and the usage of the code and artifacts is divided according to these aspects. They are ordered in terms of most-common-use first.

### Folder setup and common instructions
* &rarr; Firstly obtain the dataset, placed in the folder data_plms, at the following Google Drive link (you are required to request access thru a gmail ID): [geo_data_all](https://drive.google.com/drive/folders/13s2hzDxm6tgXqPxhVK_5248edgOAwKOb?usp=sharing)

* &rarr; Place the data elements in a folder named "data" as a sibling to the different code files.

* &rarr; The repo has a folder "resources" which also must be placed as a sibling to the different code files.

* &rarr; Inside the "resources" folder, create a folder named "transformers" and place all the transformer models that you want to try out as part of the experimentation. For e.g., create a folder "bert_base_uncased" and place its models files inside the folder. The same folder name will be used as "model_folder_name" in the commands and the code will fetch the transformer files from this folder's path.

* &rarr; Create a new python 3.7.10 environment and install dependencies in requirements.txt inside the environment. The environment should be activated to run all code in the repo.

* &rarr; For each of the aspects, the entries in the python commands marked in < and > need to be filled with appropriate values

### Aspect 1: Gauging an encoder PLM for its geography knowledge
**lm_checker_encoder_models.py**: Run the prompt probing for a specific PLM over all the generated masked sentences.

```
python lm_checker_encoder_models.py <model_folder_name> <true if the model is uncased, false otherwise> data\generated_masked_sentences\en data\lm_predictions
```

### Aspect 2: Gauging a decoder PLM for its geography knowledge
**lm_checker_decoder_models.py**: Run the prompt probing for a specific PLM over all the generated masked sentences.

```
python lm_checker_decoder_models.py <model_folder_name> data\autoregressive_masked_sentences\en data\lm_predictions
```

### Aspect 3: Application: Searching Confluence Point data
See the following repo for detailed instructions:


### Aspect 4: Application: WikiData bordering relation augmentation
See the following repo for detailed instructions:


### Aspect 5: Fine-tuning encoder PLMs for geography knowledge (available as part of the dataset)


### Aspect 6: Fine-tuning decoder PLMs for geography knowledge (available as part of the dataset)


### Aspect 7: Creating the dataset

1. **query_generator.py**: Create queries to fetch triples from WikiData related to multiple geographical entities and their relations (expressed through properties) with other entities. The command creates queries for multiple languages apart from English.
```
python query_generator.py resources/wikidata_queries
```

2. **query_runner.py**: Execute the created queries using the package SPARQLWrapper and [Wikidata query service URL](https://query.wikidata.org/sparql). This will create a file for each entity and property pair being considered with triples of the form subject_entity, property/relation, object_entity.
```
python query_runner.py resources/wikidata_queries/<query_file_path> data\wikidata_query_results\<lang>
```

3. **masked_sentence_generator.py**: Based on the templates specified for each entity and property pair, create masked sentences.

```
python masked_sentence_generator.py resources\templates\entity_type_property_templates_<lang>.txt data\wikidata_query_results\<lang> data\generated_masked_sentences\<lang>
```

### Aspect 8: Automatic evaluation of the dataset