from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_fn(model_dir):
    # Load our model from Hugging Face
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Move model to GPU
    model.to(device)

    return model, processor


def predict_fn(data, model_and_processor):
    # unpack model and tokenizer
    model, processor = model_and_processor
    
    image = data.get("inputs")
    pixel_values = processor.feature_extractor(image, return_tensors="pt").pixel_values
    task_prompt = "<s>" # start of sequence token for decoder since we are not having a user prompt
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    return prediction

