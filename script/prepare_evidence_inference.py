import json
import os

input_dir = "/root/autodl-tmp/code/multivers-main/data_train/pretrain/evidence_inference"
output_dir = "/root/autodl-tmp/code/multivers-main/data/evidence_inference_processed"
os.makedirs(output_dir, exist_ok=True)

def process_file(filename):
    entries = []
    
    with open(os.path.join(input_dir, filename), 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Start with original data
            entry = data.copy()
            
            # Fix Evidence Sets & Label if needed
            # The original data has 'evidence_sets' but sometimes label is inconsistent or missing
            # We ensure that if label is NOT NEI, evidence_sets is populated correctly.
            # However, for Evidence Inference, the original 'evidence_sets' structure is:
            # [[sent_idx, ...], [sent_idx, ...]] which ExternalReader expects.
            
            # In original data:
            # "evidence_sets": [] or [[1, 2], [5]]
            
            # We just need to ensure that we don't have (Label=SUPPORT, Evidence=[])
            # But we can't invent evidence. We trust the original data's evidence_sets.
            # The issue might be that 'label' needs to be mapped to Support/Refute/NEI?
            # ExternalReader uses data["label"] directly.
            # Original labels: "significantly increased", etc.
            # SciFactDataset maps: "REFUTES", "NOT ENOUGH INFO", "SUPPORTS".
            
            # MAPPING LOGIC
            orig_label = data.get("label", "NOT ENOUGH INFO")
            mapped_label = "NOT ENOUGH INFO"
            
            if orig_label == "significantly increased":
                mapped_label = "SUPPORTS"
            elif orig_label == "significantly decreased":
                mapped_label = "CONTRADICT"
            elif orig_label == "no significant difference":
                # This is tricky. Usually treated as CONTRADICT if claim says there is diff?
                # Or NEI?
                # Let's look at what EvidenceInferenceReader expects.
                # It doesn't have a custom label map. SciFactDataset uses standard map.
                # If we use "CONTRADICT", it maps to 0.
                mapped_label = "CONTRADICT" 
            else:
                mapped_label = orig_label # Pass through if already standard
            
            entry["label"] = mapped_label
            
            # ExternalReader expects "evidence_sets"
            # Original data has it.
            
            entries.append(entry)
    
    return entries

print("Processing train.jsonl...")
train_entries = process_file("train.jsonl")

print("Processing dev.jsonl...")
dev_entries = process_file("dev.jsonl")

# Save files to the DIRECTORY where Reader expects them
target_dir = "/root/autodl-tmp/code/multivers-main/data_train/pretrain/evidence_inference"
# Backup original
if not os.path.exists(os.path.join(target_dir, "train.jsonl.bak")):
    os.rename(os.path.join(target_dir, "train.jsonl"), os.path.join(target_dir, "train.jsonl.bak"))
    os.rename(os.path.join(target_dir, "dev.jsonl"), os.path.join(target_dir, "dev.jsonl.bak"))

print(f"Saving to {target_dir}...")

with open(os.path.join(target_dir, "train.jsonl"), 'w') as f:
    for entry in train_entries:
        f.write(json.dumps(entry) + "\n")
        
with open(os.path.join(target_dir, "dev.jsonl"), 'w') as f:
    for entry in dev_entries:
        f.write(json.dumps(entry) + "\n")

print("Done! Files updated in Reader directory.")

