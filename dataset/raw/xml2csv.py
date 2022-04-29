import json
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# bio_human_version_1.0.xml is a subset of the biosample dataset (https://ftp.ncbi.nih.gov/biosample/)
# The experiments use the biosample dataset, the copy published in Feb 12, 2020
# bio_human_version_1.0.xml is the data in package Human.1.0 of the biosample dataset

path = "bio_human_version_1.0.xml"
if os.path.exists(path):
    print("start processing "+path)

# get elements in xml
tree = ET.parse(path)
root = tree.getroot()
element_list = []
for bio_sample in root:

    element_map = {}
    bio_keys = bio_sample.attrib.keys()
    # get attributes in BioSample
    if 'submission_date' in bio_keys:
        element_map['submission_date'] = bio_sample.attrib['submission_date']
    if 'id' in bio_keys:
        element_map['id'] = bio_sample.attrib['id']
    if 'accession' in bio_keys:
        element_map['accession'] = bio_sample.attrib['accession']

    for child in bio_sample:
        # get Ids
        if child.tag == 'Ids':
            for id in child:
                if 'db_label' in id.attrib.keys():
                    element_map['sample_name'] = id.text
        # get Description
        elif child.tag == 'Description':
            for discription in child:
                if discription.tag == 'Title':
                    element_map['sample_title'] = discription.text
                elif discription.tag == 'Organism':
                    for organism in discription:
                        if organism.tag == 'OrganismName':
                            element_map['organism'] = organism.text
        # get Owner
        elif child.tag == 'Owner':
            for owner in child:
                if owner.tag == 'Name':
                    element_map['owner_name'] = owner.text
                elif owner.tag == 'Contacts':
                    for contact in owner:
                        if 'email' in contact.attrib.keys():
                            element_map['owner_email'] = contact.attrib['email']
        # get Attributes
        elif child.tag == 'Attributes':
            for attribute in child:
                if 'attribute_name' in attribute.attrib:
                    key = attribute.attrib['attribute_name']
                    key = key.replace('-', '_')
                    key = key.replace(' ', '_')
                    if key == 'biomaterialprovider':
                        key = 'biomaterial_provider'
                    elif key == 'cell_lines':
                        key = 'cell_line'
                    element_map[key.lower()] = attribute.text
    element_list.append(element_map)

# get the fields list in the bio data template
form_fields = ["id", "accession", "submission_date", "sample_name", "sample_title",
           "sample_type", "bioproject_accession", "organism", "isolate", "age", "sex", "tissue", "cell_line",
           "cell_type", "cell_subtype", "culture_collection", "dev_stage", "disease", "disease_stage", "ethnicity",
           "health_state", "karyotype", "phenotype", "population", "race", "description"]

bios = {}
for field in form_fields:
    bios[field] = []
for element_map in element_list:
    for field in form_fields:
        if field not in element_map.keys():
            bios[field].append(None)
        else:
            bios[field].append(element_map[field])

form_df = pd.DataFrame(bios)
form_df.to_csv("tmp.csv", index=False)

# normalize synonyms
path = "tmp.csv"
form_df = pd.read_csv(path)  # read the dataset
os.remove(path)
print(form_df.shape)
categories = ["sex", "tissue", "cell_type", "disease", "ethnicity", "cell_line"]
with open("synonyms.json")as cf:
    synonyms = json.load(cf)
synonyms_all = set()
synonyms_map = {}
synonyms_map_none = {}
for key in synonyms.keys():
    for elem in synonyms[key]:
        synonyms_all.add(elem)
        if elem != key:
            if key == 'n/a':
                synonyms_map_none[elem] = key
            synonyms_map[elem] = key
synonyms_all.add(np.nan)

for col in form_df.columns.tolist():
    print(col)
    values_new = []
    if col not in categories:
        values = form_df[col]
        for i in range(0, len(values)):
            val = str(values[i]).strip()
            if val in synonyms_map_none.keys():
                values_new.append(synonyms_map_none[val])
            else:
                values_new.append(val)
        form_df[col] = values_new
    else:
        values = form_df[col]
        for i in range(0, len(values)):
            val = str(values[i]).strip()
            if val in synonyms_map.keys():
                values_new.append(synonyms_map[val])
            else:
                values_new.append(val)
        form_df[col] = values_new
form_df.to_csv("tmp.csv", index=False)

# select rows that filled with mor than half categorical values
path = "tmp.csv"
form_df = pd.read_csv(path)  # read the dataset
os.remove(path)
print(form_df.shape)
form_df = form_df[(form_df['organism'] == "Homo sapiens") | (form_df['organism'] == "homo sapiens")
                  | (form_df['organism'] == "Homo Sapiens")]
form_df['organism'] = "Homo sapiens"
print(form_df.shape)

form_df["sum"] = form_df[categories].isnull().sum(axis=1)
threshold = len(categories)/2
form_df = form_df[form_df["sum"] <= threshold]
form_df = form_df.drop("sum", axis=1)
for col in categories:
    form_df = form_df[form_df[col].isin(synonyms_all)]
print(form_df[categories].shape)
print(form_df[categories])
form_df = form_df.fillna('n/a')
form_df.to_csv("./ncbi-homo-sapiens.csv", index=False)
