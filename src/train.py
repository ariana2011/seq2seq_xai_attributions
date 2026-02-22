import torch
import numpy as np
import torch.nn as nn
import transformers
from model import MarianMTModel, MarianConfigCustom
from configuration_marian import MarianConfig
from torch.utils.data import DataLoader
from data_loader import TranslationDataset
import evaluate
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from transformers import Seq2SeqTrainer, default_data_collator, DataCollatorForSeq2Seq
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, set_seed, AutoConfig, AutoModelForSeq2SeqLM
from datasets import load_dataset, Features, Array2D

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(train_loader, val_loader, model, epochs, save_path, num_steps):
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    warmup_steps = int(num_steps / 100)
    print('Warmup steps: ', warmup_steps, ' Training steps: ', num_steps)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)#torch.optim.lr_scheduler.LambdaLR()
    counter = 0
    best_vloss = 1_000_000.

    timestamp = datetime.now()
    writer = SummaryWriter('runs/de-en_{}'.format(timestamp))


    for epoch in range(epochs):
        print('Epoch:', epoch)
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_loader, optimizer, scheduler)

        running_vloss = 0.
        running_vbleu = 0.
        model.eval()

        with torch.no_grad():
            print('Validation...')
            for i, vsentence in tqdm(enumerate(val_loader)):
                vinputs, vlabels, vsrc_mask, vtrg_mask = vsentence['source'].to('cuda'), vsentence['target'].to('cuda'),\
                    vsentence['encoder_attention_mask'].to('cuda'), vsentence['decoder_attention_mask'].to('cuda')
                #vinputs, vlabels = vdata['source'].to('cuda'), vdata['target'].to('cuda')
                vout = model(input_ids=vinputs, attention_mask=vsrc_mask, decoder_input_ids=vlabels, decoder_attention_mask=vtrg_mask, labels=vlabels)
                vloss = vout.loss#criterion(vout, vlabels)
                vpredictions = model.generate(**{'input_ids': vinputs, 'attention_mask': vsrc_mask})# num_beams  = 20, max_length=128, early_stopping=True)
                vbleu = compute_metrics([vpredictions, vlabels], tokenizer)
                running_vloss += vloss
                running_vbleu += vbleu['bleu']

        avg_vloss = running_vloss / (i + 1)
        avg_vbleu = running_vbleu / (i + 1)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('BLEU valid: ', avg_vbleu)

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.add_scalar('BLEU/val', avg_vbleu, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if counter > 10:
            print('Stopped training because there is no improvement in val_loss since 10 epochs')
            return

    
def train_one_epoch(index, tb_writer, train_loader, optimizer, scheduler):
    running_loss = 0.
    last_loss = 0.

    for i, sentence in tqdm(enumerate(train_loader)):
    
        inputs, labels, src_mask, trg_mask = sentence['input_ids'], sentence['labels'], sentence['attention_mask'], sentence['attention_mask']
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=src_mask,  decoder_input_ids=labels, decoder_attention_mask=trg_mask, labels=labels)

        loss = outputs.loss #criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    loss = running_loss / (i + 1)
    tb_x = index * len(train_loader) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    return loss

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    #if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds, tokenizer):
    metric = evaluate.load("sacrebleu")# cache_dir=model_args.cache_dir)
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    #preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds.cpu()]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train_hf():
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    config = MarianConfig(max_position_embeddings=128, pad_token_id=58100)
    config.save_pretrained('configs/marian_default_config.json')
    config = config.from_json_file('configs/marian_default_config.json')
    model = MarianMTModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    data_collator = default_data_collator

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__=='__main__':
    print('Let\'s train!!') 
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--chkpoint', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--source_lang', default='de')
    parser.add_argument('--target_lang', default='en')
    parser.add_argument('--padding', default='max_length')
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--max_source_length', type=int, default=128)
    args = parser.parse_args()

    source_lang = args.source_lang
    target_lang = args.target_lang
    prefix = ""
    # Set seed before initializing model.
    set_seed(args.random_seed)

    metric = evaluate.load("sacrebleu")# cache_dir=model_args.cache_dir)
    #config = MarianConfig(max_position_embeddings=128)
    #config.save_pretrained('configs')
    #config = MarianConfig.from_json_file('configs/config.json')
    #model = MarianMTModel(config=config)

    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config.json')
    custom_model = MarianMTModel(config=custom_config)
    AutoConfig.register("custom_marian", MarianConfigCustom)
    AutoModelForSeq2SeqLM.register(MarianConfigCustom, MarianMTModel)

    custom_model = AutoModelForSeq2SeqLM.from_config(custom_config)#'checkpoints/initial_checkpoint')
    custom_model = custom_model.apply(custom_model._init_weights)

    def initialize_weights(module):

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
    model = custom_model
    model = model.apply(initialize_weights)

    model.resize_token_embeddings(len(tokenizer))

    if args.chkpoint:
        print('Loading checkpoint')
        ckpt_path = 'checkpoints/marianmt'
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')))


    model.to('cuda')
    print(model)
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    batch_size = args.batch_size
    max_target_length = args.max_target_length
    padding = args.padding

    train_dataset = TranslationDataset()
    data_files = {"train": 'data/de-en/train_filtered.json', "test": 'data/de-en/test_filtered.json', "validation": 'data/de-en/validation_filtered.json'}#train_filtered_baseline2
    features = Features({"data": Array2D(shape=(2, 2), dtype='int32')})
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')

    #https://huggingface.co/docs/datasets/use_with_pytorch
    for key, dataset in raw_datasets.items():
        dataset = dataset.with_format("torch", device='cuda')
    print(type(raw_datasets), type(train_dataset))

    column_names = raw_datasets["train"].column_names
    print(column_names)
    datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,#data_args.preprocessing_num_workers,
        remove_columns=['en', 'de'],
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )

    print(datasets, type(datasets['train']))

    # Data collator
    label_pad_token_id = -100 #if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    data_collator = default_data_collator
    print(data_collator)
    input()

    train_generator = torch.Generator().manual_seed(42)
    train_dataset = datasets['train'].with_format('torch', device='cuda')
    if args.debug:
        train_dataset = train_dataset.select(range(100))#torch.utils.data.Subset(train_dataset, range(1000))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator, collate_fn=data_collator)
    batch = next(iter(train_loader))
    print(f'{batch=}')
    print(train_dataset[0])

    val_dataset = datasets['validation'].with_format('torch', device='cuda')#TranslationDataset(split='validation')
    val_generator = torch.Generator().manual_seed(42)
    if args.debug:
        val_dataset = val_dataset.select(range(100))#torch.utils.data.Subset(val_dataset, range(100))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=val_generator)

    save_path = 'checkpoints/marianmt'
    num_epochs = 100

    num_steps = num_epochs * int(len(train_dataset) / batch_size)
    train(train_loader, val_loader, model, num_epochs, save_path, num_steps)
