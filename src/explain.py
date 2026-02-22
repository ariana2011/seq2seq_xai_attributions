"""
Using Inseq (https://inseq.org/) to get attributions for translation models. The script processes a dataset of source and target sentences, computes attributions using the specified XAI method from Inseq, and saves the results in JSON format. It supports resuming from existing files to avoid reprocessing already handled samples.
TODO: Save the attribution matrices in numpy format as well, and add CLI arguments to control this and the max matrix size to save. 

"""


import argparse
import copy as cp
import json
import os
import random

import inseq
import numpy as np
import torch
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from data_loader import TranslationDataset


os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
np.set_printoptions(threshold=np.inf)

# Ensure deterministic behavior for PyTorch (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Control parallelism for reproducibility
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

class Attribuitions:
    
    def _scaler_minmax(self, attributions):
        scaler = MinMaxScaler()
        tensor_np = attributions
        scaled_np = scaler.fit_transform(tensor_np)
        return scaled_np
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # subtract max for numerical stability
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def _softmax_to_one_hot(self, matrix):
        softmax_matrix = self._softmax(matrix)
        max_indices = np.argmax(softmax_matrix, axis=1)
        one_hot_matrix = np.zeros_like(matrix)
        one_hot_matrix[np.arange(matrix.shape[0]), max_indices] = 1
        return one_hot_matrix

    def _l2_norm(self, A):
        """
        Compute the L2 norm of a matrix A.
        """
        l2_norm = torch.norm(A, p=2, dim=-1)
        l2_norm_np = l2_norm.numpy()

        return l2_norm_np
        
    def get_attributions(self, sentence, seq_att, j):
        """
        Extract and clean the attribution matrix for a single example in a batch.

        Args:
            sentence: The current batch dict containing source/target tokens and metadata.
            seq_att:  Raw attribution array (numpy) of shape (target_len, source_len)
                      as returned by inseq's aggregate step.
            j:        Index of the example within the batch.

        Returns:
            attributions (np.ndarray): Cleaned attribution matrix with NaNs replaced by 0.
            j (int):                   The same batch index, passed through for convenience.
            att_shape (tuple):         Shape of the attribution matrix (target_len, source_len).
        """
        

        ids = cp.deepcopy(sentence['idx'])
        id =  ids.tolist()[j]        
        
        source_attributions_numpy = seq_att

        
        attributions = source_attributions_numpy
        att_shape = source_attributions_numpy.shape

        attributions = np.nan_to_num(seq_att, nan=0)

        return attributions, j , att_shape


    def _absmax_quantize(self, X):
        # Calculate scale
        """
        from https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c
        """
        scale = 127 / torch.max(torch.abs(X))

        # Quantize
        X_quant = (scale * X).round()

        # Dequantize
        X_dequant = X_quant / scale

        return X_quant.to(torch.int8), X_dequant, scale

    def _zeropoint_quantize(self, X):
        """
        from https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c
        """
        # Calculate value range (denominator)
        x_range = torch.max(X) - torch.min(X)
        x_range = 1 if x_range == 0 else x_range
        min_max_vector = torch.tensor([torch.max(X), torch.min(X)])
        # Calculate scale
        scale = 255 / x_range

        # Shift by zero-point
        zeropoint = (-scale * torch.min(X) - 128).round()

        # Scale and round the inputs
        X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

        # Dequantize
        X_dequant = (X_quant - zeropoint) / scale

        return X_quant.to(torch.int8), X_dequant, min_max_vector


class DataSave:    
    def json_object(self, attr_src, sentence, j, shape_src, attr_output = None, shape_output = None): 
        if attr_output is not None:
            result = {str(input_lang):"",str(output_lang):"", "idx": 0, "attribution": [], "shape_src": [], "target_attributions": [], "shape_target": []}
            result['target_attributions'] = attr_output.flatten().tolist()
            result['shape_target'] = shape_output
        else:       
            result = {str(input_lang):"",str(output_lang):"", "idx": 0, "attribution": [], "shape_src": []}
        ids = cp.deepcopy(sentence['idx'])
        id =  ids.tolist()[j]

        result[input_lang] = sentence[input_lang][j]
        result[output_lang] = sentence[output_lang][j]
        result['idx'] = id
        result['shape_src'] = shape_src
        
        result['attribution'] = attr_src.flatten().tolist()
        return result

    def save_json(self, fp, result, language = 'en'):        
        if language != 'en':
            json.dump(result, fp, ensure_ascii=False)
            fp.write('\n')
        else:
            json.dump(result, fp)
            fp.write('\n')


    def path_maker(self):
        """
        Create the output directory if it doesn't exist.
        """
        if args.generate_dest:
            output_dir = f'attributions/{data_folder}/{args.model}/{args.xi_method}/generated/{args.data_name}/'
            output_dir_outputseq = f'attributions/{data_folder}/{args.model}/{args.xi_method}/generated/{args.data_name}/out_seq/'
        else:
            output_dir = f'attributions/{data_folder}/{args.model}/{args.xi_method}/{args.data_name}/'
            output_dir_outputseq = f'attributions/{data_folder}/{args.model}/{args.xi_method}/{args.data_name}/out_seq/'
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir_outputseq):
            os.makedirs(output_dir_outputseq)
        return output_dir, output_dir_outputseq


    def file_remover_directory(self, file_name):
        """
        Remove the directory if it exists."""
        specific_file = file_name
        output_dir = f'attributions/{data_folder}/{args.model}/{args.xi_method}/{args.data_name}/'
        if specific_file in os.listdir(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
                    
    def file_remover(self, file_name):
        """Remove the specified file if it exists."""
        if args.generate_dest:
            output_dir = f'attributions/{data_folder}/{args.model}/{args.xi_method}/generated/{args.data_name}/'
        else:
            output_dir = f'attributions/{data_folder}/{args.model}/{args.xi_method}/{args.data_name}/'
        file_path = os.path.join(output_dir, file_name)
        if os.path.isfile(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def load_processed_keys(self, file_path: str, key: str = "idx"):
        """Load a set of already-processed values for `key` from an existing JSONL file.
        Returns an empty set if file does not exist or is unreadable.
        """
        processed = set()
        if not os.path.isfile(file_path):
            return processed
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and key in obj and obj[key] is not None:
                            processed.add(str(obj[key]))
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: could not read existing file {file_path}: {e}")
        return processed

def main(resize = None, one_diag=False, rand_uniform=False, rand_gaussian=False, one_hot=False, matrix_op = False, scaler = None, generate= None, iteraiton = 0, step_size = 100_000):

    atts = Attribuitions()
    data = DataSave()
    output_dir, output_dir_outputseq = data.path_maker()
    fp = None
    step_size = 100_000
    file_counter = args.json_num
    sample_counter = args.sample_num
    processed_key_name = args.resume_key if args.resume_key else "idx"
    processed_set = set()
    skipped_count = 0
    
    data_files = {f"{args.data_name}": f'data/{args.data}'}
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')
    
    current_file = os.path.join(output_dir, f'{args.data_name}_{file_counter}.json')

    
    if args.max_num:
        print(f"{args.max_num} samples will be processed...")
        train_dataset = raw_datasets[f"{args.data_name}"].select(range(args.sample_num, args.max_num)).with_format("torch", device='cuda')
    else:
        train_dataset = raw_datasets[f"{args.data_name}"].with_format("torch", device='cuda')
    
    print(train_dataset)
    if args.resume:
            processed_set = data.load_processed_keys(current_file, key=processed_key_name)

            processed_ids = {int(i) for i in processed_set}
            train_dataset = train_dataset.filter(
                    lambda batch: [int(i) not in processed_ids for i in batch["idx"]],
                    batched=True,
                )
    print(train_dataset)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    
    for i, sentence in tqdm(enumerate(data_loader)):
        if i % step_size == 0:
            if fp:
                fp.close()  # Close the previous file before opening a new one
            # If resuming, do not remove existing file; load processed keys from it

            if not args.resume:
                data.file_remover(f'{args.data_name}_{file_counter}.json')
                processed_set = set()
            fp = open(current_file, 'a', encoding='utf-8')
            file_counter += 1


        source = sentence[input_lang]
        dest = sentence[output_lang]#.to('cuda')

        try:

            if generate:
                print('Generating destination texts...')
                try:
                    out = inseq_model.attribute(input_texts = source, device='cuda', pretty_progress=False, attribute_target=True, show_progress=False)
                    sentence[output_lang] = out.info['generated_texts']
                    
                except ValueError as e:
                    out = inseq_model.attribute(input_texts = source, device='cuda', pretty_progress=False, attribute_target=False, show_progress=False)
                    sentence[output_lang] = out.info['generated_texts']
            else:
                
                try:
                    out = inseq_model.attribute(input_texts = source, generated_texts= dest, device='cuda', pretty_progress=False, attribute_target=True, show_progress=False)
                except ValueError as e:
                    out = inseq_model.attribute(input_texts = source, generated_texts= dest, device='cuda', pretty_progress=False, attribute_target=False, show_progress=False)
        except RuntimeError as e:
            print(f"[WARN] RuntimeError on this sample: {e} → skipping it.")
            continue

        # Re-raise other RuntimeErrors

        for j, san in enumerate(range(len(out.sequence_attributions))):
            # Skip examples already processed based on the resume key (default 'en')
            current_key_val = None
            if isinstance(sentence, dict) and processed_key_name in sentence:
                try:
                    current_key_val = sentence[processed_key_name][j]
                    # convert tensors to str if needed
                    if hasattr(current_key_val, 'cpu'):
                        current_key_val = current_key_val
                    current_key_val = str(current_key_val)
                except Exception:
                    current_key_val = None
            if current_key_val is not None and current_key_val in processed_set:
                skipped_count += 1
                continue
            
            attributions_output_seq = None
            att_output_shape = None
        
            normalized_data = out[j].aggregate(normalize=False)

            src_attr = normalized_data.source_attributions

            attributions, j ,att_source_shape = atts.get_attributions(sentence, src_attr, j)
            if att_source_shape[0] > 128 or att_source_shape[1] > 128: #If the attribution matrix is larger than 128x128, skip this example to avoid issues with large matrices. TODO: Add it it to CLI arguments.
                continue
            if normalized_data.target_attributions is not None:

                trg_attr = normalized_data.target_attributions
                attributions_output_seq, j , att_output_shape = atts.get_attributions(sentence, trg_attr, j)
                if att_output_shape[0] > 128 or att_output_shape[1] > 128: #If the attribution matrix is larger than 128x128, skip this example to avoid issues with large matrices. TODO: Add it it to CLI arguments.
                    continue

            result = data.json_object(attributions, sentence, j, att_source_shape, attr_output = attributions_output_seq, shape_output= att_output_shape)
            data.save_json(fp, result, input_lang)
            # Track as processed to avoid duplicates within the same file session
            try:
                if processed_key_name in result and result[processed_key_name] is not None:
                    processed_set.add(str(result[processed_key_name]))
            except Exception:
                pass
            sample_counter += 1
    if fp:
        fp.close()
    if skipped_count:
        print(f"Resumed: skipped {skipped_count} already-processed items (key='{processed_key_name}').")


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('--xi_method', type=str, help='XAI method to use from inseq.')
    parser.add_argument('--data', type=str, help='path to the json file.')
    parser.add_argument('--data_name', '-dn', type=str, help= 'name of the dataset [train, test, validation]')
    parser.add_argument('--data_folder','-df', type=str, help= 'de-en or fr-en or ...')
    parser.add_argument('--sample_num', '-sn', type=int, default=0, help='number of samples to process start')
    parser.add_argument('--json_num', '-jn', type=int, default=0, help='number of json to save')
    parser.add_argument('--max_num', '-mxn', type=int, default=0, help='How many samples to process')
    parser.add_argument('--batch_size', '-bs', type=int, default=20, help='Batch size')
    parser.add_argument('--save_numpy', action='store_true', help='Save numpy arrays if this flag is set')
    parser.add_argument('--one_diag', action='store_true', help='creats one diagonal matrix')
    parser.add_argument('--rand_uniform', action='store_true', help='creats random uniform matrix')
    parser.add_argument('--rand_gaussian', action='store_true', help='creats random gaussian matrix')
    parser.add_argument('--matrix_op', type=str, default='mean', help='Matrix operation to use [l2_norm, mean]')
    parser.add_argument('--scaler', type=str, default='no-scaler', help='Scaler to use [no-scaler, minmax, softmax, one_hot]')
    parser.add_argument('--generate_dest','-gd', action='store_true', help='Generate destination text if this flag is set')
    parser.add_argument('--model', type=str, help='Model to use from Huggingface')
    parser.add_argument('--resume', action='store_true', help='If set, resume writing into existing output file(s) and skip already processed items by key.')
    parser.add_argument('--resume_key', type=str, default='idx', help='Key in the JSON object to use for resume skipping (default: en).')
    parser.add_argument('--input_lang', type=str, help='Input language e.g. en', required=False)
    parser.add_argument('--output_lang', type=str, help='Output language e.g. fr', required=False)
    args = parser.parse_args()
    print(args)
    one_diag = args.one_diag
    rand_uniform = args.rand_uniform
    rand_gaussian = args.rand_gaussian
    matrix_op = args.matrix_op
    scaler = args.scaler
    generate = args.generate_dest
    data_folder = args.data_folder
    
    
    input_lang = data_folder.split('-')[0]
    output_lang = data_folder.split('-')[1]
    
    if args.model == "Marian":
        model_path = f'Helsinki-NLP/opus-mt-{input_lang}-{output_lang}'
        if args.xi_method == "layer_gradient_x_activation":
            mdl   = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
            target_layer = mdl.model.encoder.layers[5]
            inseq_model = inseq.load_model(mdl, args.xi_method, tokenizer=model_path, device='cuda', target_layer=target_layer)
        else:
            inseq_model = inseq.load_model(model_path, args.xi_method, tokenizer=model_path, device='cuda')
    if args.model == "MBart":
        model_path = f'facebook/mbart-large-50'
        if args.xi_method == "layer_gradient_x_activation":
            mdl   = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
            target_layer = mdl.model.encoder.layers[11]

            inseq_model = inseq.load_model(mdl, args.xi_method, tokenizer=model_path, device='cuda', tokenizer_kwargs={"src_lang": f"{args.input_lang}", "tgt_lang": f"{args.output_lang}"}, target_layer=target_layer)
        else:
            inseq_model = inseq.load_model(model_path, args.xi_method, tokenizer=model_path, device='cuda', tokenizer_kwargs={"src_lang": f"{args.input_lang}", "tgt_lang": f"{args.output_lang}"})

    batch_size = args.batch_size

    print(f'Explaination model: {args.xi_method}')
    print(f'Path to data: {args.data}')

    
    main(one_diag=one_diag, rand_uniform=rand_uniform, rand_gaussian=rand_gaussian, matrix_op=matrix_op, scaler = scaler, generate= generate, iteraiton = 0, step_size = 100_000)

