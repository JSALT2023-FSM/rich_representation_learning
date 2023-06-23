#!/usr/bin/env python3
"""
Tamasheq-French data processing.

Author
------
Marcely Zanon Boito 2022
"""

import json
import sys
import os
import pandas as pd
import torchaudio
from tqdm import tqdm
import csv

def write_json(output_json, results_folder, out_name):
     with open(os.path.join(results_folder, out_name+".json"), mode="w", encoding="utf-8") as output_file:
        json.dump(
                output_json,
                output_file,
                ensure_ascii=False,
                indent=2,
                separators=(",", ": "),
            )


def generate_jsons(table_path, folder_path):
    
    table = pd.read_csv(
        table_path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

    #table = pd.read_table(table_path, names = ["fileid", "translation", "split"])
    print(table.head)
    output_json = dict()
    print(f" number of lines in the table : {table.shape[0]}")
    #table = table[0:10000]
    for index, row in tqdm(table.iterrows()):
        utt_id = row["path"]
        audio_path = os.path.join(folder_path, utt_id)
        try : 
            info = torchaudio.info(audio_path) 
     
            duration = info.num_frames / info.sample_rate
            #if duration > 
            if duration > 25 : 
                print("file too long")
                continue
            transcription = row["translation"]
            if not os.path.exists(audio_path):
                continue
            #print(row)
            output_json[utt_id] = dict()
            output_json[utt_id]["path"] = audio_path
            output_json[utt_id]["trans"] = row["translation"]
            output_json[utt_id]["duration"] = duration
        except : 
            continue
    
    return output_json


def read_file(f_path):
    return [line for line in open(f_path)]


def data_proc(table_path,folder_path, output_folder, out_name):
    """

    Prepare .csv files for librimix

    Arguments:
    ----------
        dataset_folder (str) : path for the dataset github folder
        output_folder (str) : path where we save the json files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_json = generate_jsons(table_path, folder_path)

    write_json(output_json, output_folder, out_name)


if __name__=="__main__" : 
    
    table_fold = "/gpfsscratch/rech/nou/uzn19yk/covost2/splits/"
    tables = ["covost_v2.en_de.dev.tsv", "covost_v2.en_de.test.tsv", "covost_v2.en_de.train.tsv"]
    tables = [ "covost_v2.en_de.train.tsv"]
    out_names = ["dev", "test", "train"]
    out_names = ["train"]
    folder_path = "/gpfsscratch/rech/nou/uzn19yk/covost2/good_clips/"
    output_folder ="/gpfsstore/rech/nou/uzn19yk/JSALT/data/covostde-en-jsons/"
    for ind, table in enumerate(tables) : 
        table_path = os.path.join(table_fold, table)
        data_proc(table_path,folder_path, output_folder, out_names[ind])
