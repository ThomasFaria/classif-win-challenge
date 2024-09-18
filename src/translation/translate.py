def translate_batch(
    batch,
    lang_iso_3,
    tokenizer,
    model,
    device,
    max_length_encoded=512,
    max_length_decoded=512,
    forced_bos_token="eng_Latn",
):
    """
    Translates a batch of text from a source language to a target language using a tokenizer and model.

    Args:
        batch (list of str): The batch of texts to translate.
        lang_iso_3 (str): The ISO-3 language code for the source language (e.g., "fra_Latn").
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding and decoding text.
        model (transformers.PreTrainedModel): The pre-trained model used for generating translations.
        device (torch.device): The device (CPU/GPU) to use for computation.
        max_length_encoded (int, optional): The maximum length for encoding the input text. Defaults to 512.
        max_length_decoded (int, optional): The maximum length for decoding the output text. Defaults to 512.
        forced_bos_token (str, optional): The ISO-3 code of the target language for forcing the BOS token in decoding. Defaults to "eng_Latn".

    Returns:
        list of str: The translated texts.
    """
    # Set the tokenizer to the source language
    tokenizer.src_lang = lang_iso_3

    # Encode the batch of texts, with padding and truncation
    encoded_txt = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,  # We assume the most important part of the text is at the beginning
        max_length=max_length_encoded,
    ).to(device)

    # Generate translated tokens from the model, with forced BOS token to guide the target language
    generated_tokens = model.generate(
        **encoded_txt,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(forced_bos_token),
        max_length=max_length_decoded,  # Max output length based on model's training constraints
    )

    # Decode the generated tokens to get the translated text, skipping special tokens
    results = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return results
