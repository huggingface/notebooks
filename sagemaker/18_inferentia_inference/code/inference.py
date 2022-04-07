import os
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.neuron

# To use one neuron core per worker
os.environ["NEURON_RT_NUM_CORES"] = "1"

# saved weights name
AWS_NEURON_TRACED_WEIGHTS_NAME = "neuron_model.pt"


def model_fn(model_dir):
    # load tokenizer and neuron model from model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = torch.jit.load(os.path.join(model_dir, AWS_NEURON_TRACED_WEIGHTS_NAME))
    model_config = AutoConfig.from_pretrained(model_dir)

    return model, tokenizer, model_config


def predict_fn(data, model_tokenizer_model_config):
    # destruct model, tokenizer and model config
    model, tokenizer, model_config = model_tokenizer_model_config

    # create embeddings for inputs
    inputs = data.pop("inputs", data)
    embeddings = tokenizer(
        inputs,
        return_tensors="pt",
        max_length=model_config.traced_sequence_length,
        padding="max_length",
        truncation=True,
    )
    # convert to tuple for neuron model
    neuron_inputs = tuple(embeddings.values())

    # run prediciton
    with torch.no_grad():
        predictions = model(*neuron_inputs)[0]
        scores = torch.nn.Softmax(dim=1)(predictions)

    # return dictonary, which will be json serializable
    return [{"label": model_config.id2label[item.argmax().item()], "score": item.max().item()} for item in scores]
