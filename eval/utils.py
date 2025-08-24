import json

from base import postprocess_generation
from collections import Counter, defaultdict

def complete_code():
    with open("/app/linc/eval/data/Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_raw.json", "r") as f:
        gens_raw_full = json.load(f)
    with open("/app/linc/eval/data/Mistral-7B-v0.1_folio-neurosymbolic-1shot_references.json", "r") as f:
        gens_ref = json.load(f)
    with open("/app/linc/eval/data/Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json", "r") as f:
        gens_prc = json.load(f)


    gens_raw_valid = []
    gens_prc_valid = []
    for list_raw, list_prc, ref in zip(gens_raw_full, gens_prc, gens_ref):
        list_raw_valid = []
        list_prc_valid = []
        for raw, prc in zip(list_raw, list_prc):
            if ref == prc and prc != "Error" and prc == "True":
                if not raw in list_raw_valid:
                    list_raw_valid.append(raw)
                    list_prc_valid.append(prc)
        gens_raw_valid.extend(list_raw_valid)
        gens_prc_valid.extend(list_prc_valid)
  
    # print(f"Loaded {len(gens_raw_valid)} valid raw generations and {len(gens_prc_valid)} valid processed generations.")

    predict_prc = []
    for gen in gens_raw_valid:
        predict_prc.append(postprocess_generation(gen))
        
    total = len(gens_prc_valid)
    matches = sum(1 for prc, pred in zip(gens_prc_valid, predict_prc) if prc == pred)
    accuracy = matches / total if total > 0 else 0.0
    print(f"Total samples: {total}")
    print(f"Matches: {matches}")
    print(f"Accuracy: {accuracy:.4f}")

    # Group and count all unique output labels in both gens_prc_valid and predict_prc
    label_counter = Counter(predict_prc)
    print("\nLabel counts (across references and predictions):")
    for label, count in label_counter.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    complete_code()