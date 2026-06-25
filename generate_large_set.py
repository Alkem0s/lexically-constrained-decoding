import json
import os
import re
import random
import argparse
import sys
from tqdm import tqdm

# Adjust path to import from project root
WORKSPACE_DIR = "/mnt/c/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Machine Translation/lexically-constrained-decoding"
sys.path.append(WORKSPACE_DIR)

import config
import model_loader
import decoding

# Stopwords lists to filter out particles, pronouns, conjunctions
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in", "on", "at", 
    "to", "for", "with", "by", "about", "is", "was", "were", "be", "been", "have", 
    "has", "had", "this", "that", "it", "they", "he", "she", "we", "you", "i",
    "are", "from", "which", "who", "whom", "whose", "as", "into", "through"
}

TR_STOPWORDS = {
    "ve", "veya", "ama", "fakat", "lakin", "ancak", "ise", "ile", "bir", "bu", 
    "o", "şu", "her", "hepsi", "tüm", "için", "gibi", "kadar", "da", "de", 
    "mi", "mu", "mü", "mı", "en", "daha", "çok", "var", "yok", "ki", "kendi",
    "biri", "şey", "herkes", "hiç", "hiçbir", "yine", "yani", "çünkü"
}

def clean_words(text, stopwords, min_len=4):
    """Lowercases, removes punctuation, filters stopwords and short tokens."""
    text = text.lower()
    # Find all alphabetical words
    words = re.findall(r'\b[a-zA-ZçıüşöğâîûÇIÜŞÖĞÂÎÛ]+\b', text)
    filtered = {w for w in words if w not in stopwords and len(w) >= min_len}
    return filtered

def is_morphological_overlap(w1, w2):
    w1 = w1.lower()
    w2 = w2.lower()
    min_len = min(len(w1), len(w2))
    if min_len >= 3:
        if w1[:3] == w2[:3]:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Download FLORES-200 and generate constrained test cases.")
    parser.add_argument("--limit", type=int, default=250, help="Number of sentence pairs per direction (total samples = 2 * limit)")
    parser.add_argument("--output", type=str, default="test_cases_eval_large.json", help="Output JSON filename")
    args = parser.parse_args()

    print(f"=== Generating Large Evaluation Test Set ({args.limit} cases per direction) ===")
    
    print("Downloading Tatoeba Parallel Corpus via huggingface_hub...")
    try:
        from huggingface_hub import hf_hub_download
        import csv
        file_path = hf_hub_download(
            repo_id="Helsinki-NLP/tatoeba_mt", 
            filename="test/tatoeba-test.eng-tur.tsv", 
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Error downloading Tatoeba dataset: {e}")
        raise e

    print("Successfully downloaded. Aligning sentences...")
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                dataset.append({
                    "sentence_eng_Latn": row[2].strip(),
                    "sentence_tur_Latn": row[3].strip()
                })
        
    limit = min(args.limit, len(dataset))
    sub_dataset = dataset[:limit]
    print(f"Using {limit} sentence pairs for constraint extraction.")

    print("Loading models...")
    en_tr_model = model_loader.load_en_tr()
    tr_en_model = model_loader.load_tr_en()

    random.seed(42)

    en_tr_cases = []
    tr_en_cases = []

    print("\n--- Processing EN -> TR Cases ---")
    for i, item in enumerate(tqdm(sub_dataset, desc="EN-TR")):
        en_src = item["sentence_eng_Latn"].strip()
        tr_ref = item["sentence_tur_Latn"].strip()

        # Get baseline translation
        tr_base, _ = decoding.unconstrained(en_tr_model, en_src)

        # Extract tokens
        ref_tokens = clean_words(tr_ref, TR_STOPWORDS)
        base_tokens = clean_words(tr_base, TR_STOPWORDS)

        required = []
        difficulty = "easy"
        is_hard_case = (i % 2 == 0) # 50-50 mix of easy/hard target settings

        if is_hard_case:
            # Hard Case: Target reference words that are MISSING from the baseline output
            missing = sorted(list(ref_tokens - base_tokens), key=len, reverse=True)
            if missing:
                required = missing[:min(2, len(missing))]
                difficulty = "hard"
            else:
                # Fallback to easy case if no missing words exist
                common = sorted(list(ref_tokens & base_tokens), key=len, reverse=True)
                if common:
                    required = [common[0]]
                    difficulty = "easy"
        else:
            # Easy Case: Target reference words that are PRESENT in the baseline output
            common = sorted(list(ref_tokens & base_tokens), key=len, reverse=True)
            if common:
                required = [common[0]]
                difficulty = "easy"
            else:
                # Fallback to hard case if baseline and reference have 0 overlap
                missing = sorted(list(ref_tokens - base_tokens), key=len, reverse=True)
                if missing:
                    required = missing[:min(2, len(missing))]
                    difficulty = "hard"

        # Exclusion (Forbidden) Constraints: baseline words not present in reference
        extra = sorted(list(base_tokens - ref_tokens), key=len, reverse=True)
        forbidden = []
        for ext_w in extra:
            overlap = False
            for req_w in required:
                if is_morphological_overlap(ext_w, req_w):
                    overlap = True
                    break
            if not overlap:
                forbidden.append(ext_w)
                if len(forbidden) >= 2:
                    break

        en_tr_cases.append({
            "source": en_src,
            "direction": "EN→TR",
            "forbidden_words": forbidden,
            "required_words": required,
            "penalty_words": forbidden,
            "reward_words": required,
            "difficulty": difficulty,
            "comment": f"FLORES-200 dev index {i} ({difficulty} inclusion)"
        })

    print("\n--- Processing TR -> EN Cases ---")
    for i, item in enumerate(tqdm(sub_dataset, desc="TR-EN")):
        tr_src = item["sentence_tur_Latn"].strip()
        en_ref = item["sentence_eng_Latn"].strip()

        # Get baseline translation
        en_base, _ = decoding.unconstrained(tr_en_model, tr_src)

        # Extract tokens
        ref_tokens = clean_words(en_ref, EN_STOPWORDS)
        base_tokens = clean_words(en_base, EN_STOPWORDS)

        required = []
        difficulty = "easy"
        is_hard_case = (i % 2 == 0)

        if is_hard_case:
            missing = sorted(list(ref_tokens - base_tokens), key=len, reverse=True)
            if missing:
                required = missing[:min(2, len(missing))]
                difficulty = "hard"
            else:
                common = sorted(list(ref_tokens & base_tokens), key=len, reverse=True)
                if common:
                    required = [common[0]]
                    difficulty = "easy"
        else:
            common = sorted(list(ref_tokens & base_tokens), key=len, reverse=True)
            if common:
                required = [common[0]]
                difficulty = "easy"
            else:
                missing = sorted(list(ref_tokens - base_tokens), key=len, reverse=True)
                if missing:
                    required = missing[:min(2, len(missing))]
                    difficulty = "hard"

        # Exclusion (Forbidden) Constraints
        extra = sorted(list(base_tokens - ref_tokens), key=len, reverse=True)
        forbidden = []
        for ext_w in extra:
            overlap = False
            for req_w in required:
                if is_morphological_overlap(ext_w, req_w):
                    overlap = True
                    break
            if not overlap:
                forbidden.append(ext_w)
                if len(forbidden) >= 2:
                    break

        tr_en_cases.append({
            "source": tr_src,
            "direction": "TR→EN",
            "forbidden_words": forbidden,
            "required_words": required,
            "penalty_words": forbidden,
            "reward_words": required,
            "difficulty": difficulty,
            "comment": f"FLORES-200 dev index {i} ({difficulty} inclusion)"
        })

    # Save to JSON
    output_data = {
        "EN_TR": en_tr_cases,
        "TR_EN": tr_en_cases
    }

    out_path = os.path.join(WORKSPACE_DIR, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    en_tr_easy = sum(1 for c in en_tr_cases if c["difficulty"] == "easy")
    en_tr_hard = sum(1 for c in en_tr_cases if c["difficulty"] == "hard")
    tr_en_easy = sum(1 for c in tr_en_cases if c["difficulty"] == "easy")
    tr_en_hard = sum(1 for c in tr_en_cases if c["difficulty"] == "hard")

    print(f"\nSuccessfully generated {len(en_tr_cases) + len(tr_en_cases)} cases!")
    print(f"  EN→TR : {len(en_tr_cases)} cases ({en_tr_easy} easy, {en_tr_hard} hard)")
    print(f"  TR→EN : {len(tr_en_cases)} cases ({tr_en_easy} easy, {tr_en_hard} hard)")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()