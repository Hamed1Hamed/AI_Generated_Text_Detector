# -*- coding: utf-8 -*-

# this approach is Subword Embeddings approach.
import logging
from transformers import AutoModel, AutoTokenizer
import torch

logging.basicConfig(filename='model_comparisons.log', level=logging.INFO, filemode='a',
                    format='- %(message)s', encoding='utf-8-sig')


def compare_models_tokenization_embeddings(model_names, sentence):
    # Log and print the input sentence
    logging.info(f"Input Sentence: {sentence}")
    print(f"Input Sentence: {sentence}")

    results = {}

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        embeddings = outputs.last_hidden_state.squeeze()

        results[model_name] = {
            "tokens": tokens,
            "embeddings_shape": embeddings.shape
        }

        logging.info(f"Model: {model_name}")
        print(f"Model: {model_name}")
        logging.info("Tokens: " + ", ".join(tokens))
        print("Tokens: " + ", ".join(tokens))

        for token, emb in zip(tokens, embeddings):
            emb_list = emb.tolist()
            emb_str = ', '.join(f"{x:.4f}" for x in emb_list)
            logging.info(f"Token: {token}, Embedding: [{emb_str}]")
            print(f"Token: {token}, Embedding: [{emb_str}]")

        logging.info("Embeddings Shape: " + str(embeddings.shape))
        print("Embeddings Shape: " + str(embeddings.shape))
        logging.info("-" * 30)
        print("-" * 30)

    logging.info("#" * 60)
    print("#" * 60)

    return results


model_names = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "aubmindlab/bert-base-arabertv2",
    "aubmindlab/araelectra-base-discriminator"
]
sentence = "عُلِم"
results = compare_models_tokenization_embeddings(model_names, sentence)

print("Results have been logged to 'model_comparisons.log'")
