
import torch
from transformers import MarianMTModel

import torch
import transformers

from model import MarianMTModel
from configuration_marian import MarianConfig
from transformers import AutoTokenizer


if __name__ == '__main__':
    # Load tokenizer and model configuration for MarianMT
    config = MarianConfig(max_position_embeddings=128, pad_token_id=58100)
    config = config.from_json_file('configs/config.json')

    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    model = MarianMTModel(config).to('cpu')
    # model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
    
    # Load your custom trained checkpoint
    ckpt_path = 'checkpoints/marianmt'
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=True))

    # Example German sentence
    # german_sentence = 'Deine Habgier wird noch dein Tod sein.'#'wu bist du?'
    # german_sentence = 'where are you?'
    german_sentence = 'Ruf mich an, wenn du draußen bist.'
    # Tokenize the input sentence (returns a dictionary)
    example = tokenizer(german_sentence, return_tensors='pt')
    # input()
    translated_output = model.generate(**example, num_beams  = 1, max_length=128, early_stopping=True)
    # Decode the output to human-readable text
    decoded_translation = tokenizer.decode(translated_output[0])#, skip_special_tokens=False)

    # Print the translation
    print(decoded_translation)
