import subprocess
subprocess.call(["pip", "install", "transformers==4.17.0"])
import os
import json
from transformers import GPT2Tokenizer, TextGenerationPipeline, GPT2LMHeadModel

def model_fn(model_dir):
    """
    Load the model for inference
    """

    # Load GPT2 tokenizer from disk.
    vocab_path = os.path.join(model_dir, 'model/vocab.json')
    merges_path = os.path.join(model_dir, 'model/merges.txt')
    
    tokenizer = GPT2Tokenizer(vocab_file=vocab_path,
                              merges_file=merges_path)

    # Load GPT2 model from disk.
    model_path = os.path.join(model_dir, 'model/')
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    return model.__call__(input_data)

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    return str(prediction)