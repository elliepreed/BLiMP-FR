import os
import argparse
import pandas as pd
import csv
import torch
from minicons import scorer

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing .csv files')
parser.add_argument('--model_name', type=str, required=True, help='Model name to evaluate')
parser.add_argument('--results_dir', type=str, default='model_scores', help='Output folder for results')
args = parser.parse_args()

print("🔹 Parsed arguments:", args)

def load_sentences(filepath):
    print(f"🔹 Loading sentences from {filepath}")
    sentence_pairs = []
    df = pd.read_csv(filepath)  # changed from read_parquet to read_csv
    for _, row in df.iterrows():
        sentence_pairs.append([row['sentence_good'], row['sentence_bad']])
    print(f"🔹 Loaded {len(sentence_pairs)} sentence pairs from {filepath}")
    return sentence_pairs

def compute_score(data, model, mode='ilm'):
    if mode == 'ilm':
        score = model.sequence_score(data, reduction=lambda x: x.sum(0).item())
    elif mode == 'mlm':
        score = model.sequence_score(data, reduction=lambda x: x.sum(0).item(), PLL_metric='within_word_l2r')
    return score

def process_files(model, mode, model_name, output_folder):
    # Updated to your CSV filenames instead of parquet files
    file_names = [
       'augmented_adjective_noun_agreement.csv',
       'augmented_anaphor_agreement.csv',
       'augmented_auxiliary_agreement.csv',
       'augmented_binding.csv',
       'augmented_clitic_placement.csv',
       'augmented_determiners.csv',
       'augmented_negation.csv',
       'augmented_subjunctive.csv'
    ]

    os.makedirs(output_folder, exist_ok=True)

    for file_name in file_names:
        try:
            full_path = os.path.join(args.data_dir, file_name)
            pairs = load_sentences(full_path)
            results = []
            total_difference = 0
            correct_count = 0

            for idx, pair in enumerate(pairs):
                if idx % 50 == 0:
                    print(f"🔹 Scoring sentence pair {idx + 1}/{len(pairs)} in {file_name}")
                score = compute_score(pair, model, mode)
                results.append({
                    'good_sentence': pair[0],
                    'bad_sentence': pair[1],
                    'good_score': score[0],
                    'bad_score': score[1],
                    'difference': score[0] - score[1],
                    'correct': score[0] > score[1]
                })
                if score[0] > score[1]:
                    correct_count += 1
                total_difference += score[0] - score[1]

            mean_difference = total_difference / len(pairs)
            accuracy = correct_count / len(pairs)

            output_file = os.path.join(
                output_folder,
                f"{model_name.replace('/', '_')}_{file_name}"
            )
            print(f"🔹 Writing results to {output_file}")
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            print(f"✅ Processed {file_name}:")
            print(f"  → Mean difference: {mean_difference:.4f}")
            print(f"  → Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")

    print("✅ Finished processing all files.")

# Main execution
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mode = 'ilm'
print(f"🔹 Using device: {device}")
model = scorer.IncrementalLMScorer(args.model_name, device)

process_files(
    model=model,
    mode=mode,
    model_name=args.model_name,
    output_folder=args.results_dir
)
