import json
import yaml
import argparse
import os

def read_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict

def write_json(output_json, results_folder, out_name):
    with open(os.path.join(results_folder, out_name+".json"), mode="w", encoding="utf-8") as output_file:
        json.dump(
                output_json,
                output_file,
                ensure_ascii=False,
                indent=2,
                separators=(",", ": "),
            )

def read_transript(transcript_path):
    with open(transcript_path, "r") as transcript_file:
        transcript = [
            line.strip() for line in transcript_file
        ]
    return transcript

def prepare_split(mustc_folder, output_folder, langpair, split):
    yaml_path = os.path.join(mustc_folder, langpair, "data", split, "txt", f"{split}.yaml")
    yaml_dict = read_yaml(yaml_path)

    src_lang = langpair.split("-")[0]
    transcript_path = os.path.join(mustc_folder, langpair, "data", split, "txt", f"{split}.{src_lang}")
    src_transcripts = read_transript(transcript_path)
    
    tgt_lang = langpair.split("-")[1]
    transcript_path = os.path.join(mustc_folder, langpair, "data", split, "txt", f"{split}.{tgt_lang}")
    tgt_transcripts = read_transript(transcript_path)

    data = {}
    for idx, (rec_data, src_transcript, tgt_transcript) in enumerate(zip(yaml_dict, src_transcripts, tgt_transcripts)):
        item = {
            "path": os.path.join(mustc_folder, langpair, "data", split, "wav", rec_data["wav"]),
            "trans": tgt_transcript,
            "sentence": src_transcript,
            "start": rec_data["offset"],
            "duration": rec_data["duration"],
        }
        idx += 1
        idx = rec_data["wav"].replace(".wav", f"_{idx:07d}.wav")
        data[idx] = item
    
    write_json(data, output_folder, split)

def main(mustc_folder, output_folder, langpair, splits):
    for split in splits:
        prepare_split(mustc_folder, output_folder, langpair, split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mustc_folder", type=str, help="Path to the MuST-C folder")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    parser.add_argument("langpair", type=str, help="Language pair")
    parser.add_argument("--splits", nargs="+", type=str, default=["train", "dev", "tst-COMMON", "tst-HE"], help="Splits to prepare")

    args = parser.parse_args()
    main(args.mustc_folder, args.output_folder, args.langpair, args.splits)