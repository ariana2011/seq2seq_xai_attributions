import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import random
import numpy as np
import tqdm
import glob
import json
import sys
import os

sys.path.insert(0,os.path.abspath('../'))

import attributor.TargetSourceAttributor  as TargetSourceAttributor
#import attributor.SourceAttributorPylon  as SourceAttributorPylon
#import attributor.SourceAttributorTwins as SourceAttributorTwins
#import attributor.SourceAttributorTwinsBerts as SourceAttributorTwinsBerts
#import attributor.SourceAttributorOuroboros as SourceAttributorOuroboros
#import attributor.SourceAttributorOuroborosHalt as SourceAttributorOuroborosHalt
#import attributor.LengthPredictorConver as LengthPredictorConver
#import attributor.LengthPredictorTranser as LengthPredictorTranser


class DefaultConfig:
    model = 'TargetSourceAttributor'
    n_window = 128
    n_embed = 512
    n_heads = 8
    n_vocabulary = 59514
    pad_token_id = 59513
    n_batch = 128
    epochs = 20


# Define a custom dataset class for line-by-line JSON files
class JSONLineDataset(Dataset):
    def __init__(self, json_file, tokenizer : AutoTokenizer , pair,  transform=None, second_json_file=None):
        self.data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                self.data.append(json.loads(line.strip()))  # Load each line as a JSON object
        if second_json_file:
            with open(second_json_file, "r", encoding="utf-8") as f:
                for line in tqdm.tqdm(f):
                    self.data.append(json.loads(line.strip()))
        
        self.src_lang = pair.split('-')[0]
        self.tgt_lang = pair.split('-')[1]
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_src, y_src = self.data[idx]['shape_src']
        tokens_dict = self.tokenizer(text = self.data[idx][self.src_lang], text_target = self.data[idx][self.tgt_lang],
                                      return_tensors='pt', padding='max_length', max_length=args.n_window, truncation=True)
        x = tokens_dict['input_ids'].squeeze(0)
        y = tokens_dict['labels'].squeeze(0)

        attribution= torch.tensor(self.data[idx]["attribution"], dtype=torch.float32).reshape(x_src, y_src)
        
        for j in range(y_src):
            col = attribution[:, j]

            # 1) make non-negative
            if (col < 0).any():
                shift = col.min()
                col = (col - shift).clamp_min(0.0)

            # 2) handle degenerate columns (all zeros or nearly)
            col_sum = col.sum()
            if col_sum <= 1e-8:
                col = torch.full_like(col, 1.0 / x_src)
            else:
                col = col / col_sum  # now sum ≈ 1

            attribution[:, j] = col

        #padding to full window
        attribution = F.pad( attribution, (0, args.n_window - y_src, 0, args.n_window - x_src),
            "constant",1e-10)

        # masks
        source_mask = (x != self.tokenizer.pad_token_id).long()
        target_mask = (y != self.tokenizer.pad_token_id).long()

        if not(source_mask.sum() == x_src and target_mask.sum() == y_src):
            print(f"Source mask sum {source_mask.sum()} != x_src {x_src} or Target mask sum {target_mask.sum()} != y_src {y_src}")

        return y, x, target_mask, source_mask, attribution
    

# dataset without target tokens specialized for SourceAttributor    
class JsonLineDatasetWithoutTargetTokens(JSONLineDataset):

    def __getitem__(self, idx):
        _, x, target_mask, source_mask, attribution = super().__getitem__(idx)
        return x, source_mask, target_mask.sum(-1), attribution  # Return only source, source_mask, target_length, and attributions


# Fix all seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#refactored diagnostic code into a function
def compare_attributions(model, dataset, idx):
    with torch.no_grad():
        plt.close('all')
        inp = [i.unsqueeze(0).to('cuda') for i in dataset[idx]]
        out, loss_orig, _ = model(*inp)

        orig_data = inp[-1].reshape((128, 128)).cpu()
        orig_data = orig_data.transpose(-1,-2)
        orig_data2 = orig_data[:20, :20]  # Limit to 20x20 for visualization
        plt.figure(figsize=(10, 10))
        sns.heatmap(orig_data2, cmap='viridis', square=True, cbar=True, xticklabels=False, yticklabels=False)
        plt.savefig('p_new1.png')

        out_data = out.reshape((128, 128)).detach().cpu()
        out_data2 = out_data[:20, :20]
        plt.figure(figsize=(10, 10))
        sns.heatmap(out_data2, cmap='viridis', square=True, cbar=True, xticklabels=False, yticklabels=False)
        plt.savefig('p_new2.png')

        #calculate kl divergence
        source_mask = inp[3].unsqueeze(-1).expand(-1, -1, 128)
        smooth_out = out_data.clone().to('cuda') * source_mask + 1e-10
        smooth_orig = orig_data.float().clone().to('cuda') * source_mask + 1e-10
        smooth_out = smooth_out / smooth_out.sum(dim=-1, keepdim=True)
        smooth_orig = smooth_orig / smooth_orig.sum(dim=-1, keepdim=True)
        kl_div = (smooth_orig * (torch.log(smooth_orig) - torch.log(smooth_out))).sum(dim=-1)
        target_mask = inp[2]
        masked_kl_div = kl_div * target_mask
        loss = masked_kl_div.sum() / target_mask.sum()
        print(f"KL Divergence on sample {idx}: {loss.item()}")
        print(f"{loss_orig.item()}")

# code for evaluation was move to a dedicated function
def evaluate_model(valtest_loader, record_val = False, total_train_loss = 0, epoch = -1, do_stuff=False):

      # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    total_additional_loss = 0
    additional_loss_name = 'Undefined'
    print("\nStarting evaluation...")
    loss_history = []
    bonuses = dict()
    all_stats = dict()
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(valtest_loader)):
            #target, source, target_mask, source_mask, attributions = [item.to(device) for item in data]
            
            output, loss, dlc = model(*[item.to(device) for item in data], do_stuff=do_stuff)
            loss_history.append(loss.item())
            total_val_loss += loss.item()
            if dlc is not None:
                name, add_loss = dlc[:2]
                total_additional_loss += add_loss.item()
                additional_loss_name = name
                bonus = dlc[2]
                if bonus is not None:
                    for k, v in bonus.items():
                        if k not in bonuses:
                            bonuses[k] = []
                        bonuses[k].append(v)

            if record_val:
                val_losses.append(loss.item())
    
    epoch_train_loss = total_train_loss / len(train_loader)
    epoch_valtest_mean_loss = total_val_loss / len(valtest_loader)
    if do_stuff:
        epoch_valtest_median = np.median(loss_history)
        epoch_valtest_std = np.std(loss_history)
        bonuses_avg = {f'{k}_mean': sum(v) / len(v) for k, v in bonuses.items()}
        bonuses_median = {f'{k}_median': np.median(v) for k, v in bonuses.items()}
        bonuses_std = {f'{k}_std': np.std(v) for k, v in bonuses.items()}
        all_stats = {'mean_kl_divergence': epoch_valtest_mean_loss,
                    'median_kl_divergence': epoch_valtest_median,
                    'std_kl_divergence': epoch_valtest_std,
                    **bonuses_avg,
                    **bonuses_median,
                    **bonuses_std}
    epoch_additional_loss = total_additional_loss / len(valtest_loader) if total_additional_loss > 0 else 0
    print(f"Epoch {epoch+1}/{args.epochs} complete | Train Loss: {epoch_train_loss:.6f} | Val/Test Loss: {epoch_valtest_mean_loss:.6f} | {additional_loss_name}: {epoch_additional_loss:.6f}")
    print("-" * 60)
    model.train()
    return epoch_valtest_mean_loss, all_stats

mbart_mapping = {
    'fr': 'fr_XX',
    'en': 'en_XX',
    'de': 'de_DE',
    'ar': 'ar_AR'
}

#parse command line arguments
parser = argparse.ArgumentParser(description="Train a Transformer-based attributor model.")
parser.add_argument('--model', type=str, default = DefaultConfig.model, help='choose model type')
                     #choices=['TargetSourceAttributor', 'SourceAttributorPylon', 'SourceAttributorTwinsBerts',
                               #'SourceAttributorTwins', 'SourceAttributorOuroboros', 'SourceAttributorOuroborosHalt',
                               #'LengthPredictorConver', 'LengthPredictorTranser'], help='choose model type')
parser.add_argument('--n_window', type=int, default=DefaultConfig.n_window, help='Size of the input window')
parser.add_argument('--n_embed', type=int, default=DefaultConfig.n_embed, help='Size of the embedding layer')
parser.add_argument('--n_heads', type=int, default=DefaultConfig.n_heads, help='Number of attention heads')
parser.add_argument('--n_vocabulary', type=int, default=DefaultConfig.n_vocabulary, help='Vocabulary size')
parser.add_argument('--n_batch', type=int, default=DefaultConfig.n_batch, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=DefaultConfig.epochs, help='Number of training epochs')
parser.add_argument('--from_pretrained', action='store_true', help='Path to a pretrained model to load')
parser.add_argument('--eval', action='store_true', help='Run evaluation only without training')
parser.add_argument('--split_ratio', type=str, default='90/5/5', help='Train/val/test split ratio, e.g., 80/10/10')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience based on validation loss')
parser.add_argument('--dataset_path', type=str, default='/mnt/8tera/gorilla/wmt-xai/attributions/fr-en/attention/train/',help='Path to the training dataset JSON file')
parser.add_argument('--run_name', type=str, default='', help='Name for the training run/model saving')
parser.add_argument('--pair', type=str, default='', help='Language pair for the dataset, e.g., "fr-en"')
parser.add_argument('--tokenizer', type=str, default='marian', help='type of the tokenizer deonoted by the model name')
parser.add_argument('--generated', type=str, default='human', help='type of the tokenizer deonoted by the model name')

args = parser.parse_args()

match args.model.split('_')[0]:  # Use only the prefix to match
    case 'TargetSourceAttributor':
        model_class = TargetSourceAttributor.TargetSourceAttributor
        dataset_class = JSONLineDataset
    # case 'SourceAttributorPylon':
    #     model_class = SourceAttributorPylon.SourceAttributorPylon
    #     dataset_class = JsonLineDatasetWithoutTargetTokens
    # case 'SourceAttributorTwins':
    #     model_class = SourceAttributorTwins.SourceAttributorTwins
    #     dataset_class = JSONLineDataset
    # case 'SourceAttributorTwinsBerts':
    #     model_class = SourceAttributorTwinsBerts.SourceAttributorTwinsBerts
    #     dataset_class = JSONLineDataset
    # case 'SourceAttributorOuroboros':
    #     model_class = SourceAttributorOuroboros.SourceAttributorOuroboros
    #     dataset_class = JsonLineDatasetWithoutTargetTokens
    # case 'SourceAttributorOuroborosHalt':
    #     model_class = SourceAttributorOuroborosHalt.SourceAttributorOuroborosHalt
    #     dataset_class = JsonLineDatasetWithoutTargetTokens
    # case 'LengthPredictorConver':
    #     model_class = LengthPredictorConver.LengthPredictorConver
    #     dataset_class = JsonLineDatasetWithoutTargetTokens
    # case 'LengthPredictorTranser':
    #     model_class = LengthPredictorTranser.LengthPredictorTranser
    #     dataset_class = JsonLineDatasetWithoutTargetTokens

# Load dataset from line-by-line JSON files
args.dataset_path = args.dataset_path.rstrip('/')
search_dataset_json = args.dataset_path + "/*.json"
dataset_files = glob.glob(search_dataset_json)
if not dataset_files:
    raise FileNotFoundError(f"No JSON files found in the specified dataset path: {args.dataset_path}")
else:
    dataset_path = dataset_files[0]

src_lang = args.pair.split('-')[0]
tgt_lang = args.pair.split('-')[1]
if args.tokenizer == 'marian':
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}", force_download=True, use_fast=True)
    start_token_id = tokenizer.pad_token_id
elif args.tokenizer == 'mbart':
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", force_download=True, use_fast = True)
    tokenizer.src_lang = mbart_mapping[src_lang]
    tokenizer.tgt_lang = mbart_mapping[tgt_lang]
    start_token_id = tokenizer.convert_tokens_to_ids({mbart_mapping[tgt_lang]})[0]


train_dataset = dataset_class(dataset_path, tokenizer=tokenizer, pair=args.pair)

# Split dataset into training and validation sets
train_raio, val_ratio, test_ratio = map(int, args.split_ratio.split('/'))
total_ratio = train_raio + val_ratio + test_ratio
train_size = int((train_raio / total_ratio) * len(train_dataset))
val_size = int((val_ratio / total_ratio) * len(train_dataset))
test_size = len(train_dataset) - train_size - val_size

torch.manual_seed(42)  # Ensure reproducibility for dataset split
train_dataset, val_dataset, test_dataset = random_split(
    train_dataset, [train_size, val_size, test_size]
)
print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.n_batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.n_batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.n_batch, shuffle=False)

full_name = f"{args.model}_{args.run_name}_{args.pair}_{args.tokenizer}_OVERKILL"

# Initialize model, loss function, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.from_pretrained:
    model = model_class.load_model(full_name, device)
else:
    config = TargetSourceAttributor.TAConfig(
        n_window=args.n_window,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_vocabulary=tokenizer.vocab_size,
        pad_token_id=start_token_id)
    model = model_class(config).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

if args.eval:
    print("Running evaluation only on the test dataset...")
    evaluate_model(test_loader, record_val=False, do_stuff=True)  # Evaluate without training
    sys.exit(0)

# Training loop
print_every = 1
total_train_loss = 0
patience_counter = 0
train_losses = []  # Store training loss values
val_losses = []  # Store validation loss values
val_loss_epochs = []  # Store validation loss per epoch
best_val_loss = float('inf')  # Initialize best validation loss

for epoch in range(args.epochs):
    total_train_loss = 0
    additional_loss_name = 'Undefined'
    additional_loss = 0
    running_loss = 0.0
    model.train()
    batch_count = len(train_loader)
    
    print(f"Epoch {epoch+1}/{args.epochs}")
    print("-" * 40)

    bar = tqdm.tqdm(train_loader, file =sys.stdout)
    for batch_idx, data in enumerate(bar):

        output, loss, dlc = model(*[item.to(device) for item in data])
        
        if dlc is not None:
            additional_loss_name, additional_loss = dlc[:2]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        total_train_loss += loss.item() 
        train_losses.append(loss.item())  # Store loss per step
        
        # Print statistics
        if (batch_idx + 1) % print_every == 0:
            avg_loss = running_loss / print_every
            bar.set_description(f"Batch {batch_idx+1}/{batch_count}| Loss: {loss:.6f} | Avg Loss: {avg_loss:.6f} | {additional_loss_name}: {additional_loss:.6f}")
            running_loss = 0.0

    val_loss, _ = evaluate_model(val_loader, True, total_train_loss, epoch)  # Evaluate after each epoch
    val_loss_epochs.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_model(name=full_name)
        print(f"New best validation loss: {best_val_loss:.6f}. Model saved as {full_name}.pt")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

print("Evaluation on the test dataset...")
model = model_class.load_model(full_name, device)  # Load the best model
test_loss, all_of_them = evaluate_model(test_loader, record_val=False, do_stuff=True)  # Final evaluation on test set
#save all statistics to json
with open(f"stats_{full_name}.json", "w") as f:
    json.dump(all_of_them, f, indent=4)

# Save loss values
with open(f"train_losses_{full_name}.txt", "w") as f:
    for loss in train_losses:
        f.write(f"{loss}\n")

with open(f"val_losses_{full_name}.txt", "w") as f:
    for loss in val_losses:
        f.write(f"{loss}\n")

with open(f"val_loss_epochs_{full_name}.csv", "w") as f:
    f.write("epoch,val_loss\n")
    for epoch_idx, loss in enumerate(val_loss_epochs):
        f.write(f"{epoch_idx+1},{loss}\n")

print("Training complete!")
#compare_attributions(model, val_dataset, 134)

