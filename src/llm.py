"""Loading the language model and generating text with it.

Load the model once in your notebook, then pass `model` and `processor`
into the other functions.
"""

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

DEFAULT_MODEL_NAME = "google/medgemma-4b-it"


def load_model(model_name=DEFAULT_MODEL_NAME):
    """Loads the model.

    Returns (model, processor).
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"{model_name} loaded successfully.")
    return model, processor


def generate_text(model, processor, messages, max_new_tokens=70):
    """Send a list of chat messages to the model and return its reply as text"""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Keep only the newly generated tokens (drop the prompt we sent in).
    answer_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = processor.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip()