from transformers import PreTrainedTokenizer


def convert_userprompt_transformers(tokenizer:PreTrainedTokenizer, user_prompt, add_generation_prompt=False):
    """
    Converts user-entered prompts into a conversation form by chat template though transformers

    Examples:
        tokenizer = AutoTokenizer.from_pretrained('./llama3-chat')
        print(convert_userprompt_transformers(tokenizer, user_prompt="Who are you?"))
        >>> '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|>'
    """
    content = tokenizer.apply_chat_template([{"role": "user", "content":user_prompt}], tokenize=False, add_generation_prompt=add_generation_prompt)
    return content



def skip_special_tokens_transformers(tokenizer:PreTrainedTokenizer, output):
    """
    Removing special tokens from vllm output though transformers.

    Examples:
        tokenizer = AutoTokenizer.from_pretrained('./llama3-chat')
        user_query = convert_userprompt_transformers(tokenizer, user_prompt="What is 121 times 10?")
        llm = LLM(model="./llama3-chat")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
        output = llm.generate(user_query, sampling_params)
        output_text = output[0].outputs[0].text
        print(f"output_text: {output_text}")
        
        >>> output_text: <|start_header_id|>assistant<|end_header_id|>

            Easy one!

            121 times 10 is 1,210.
            
        print(f"clean_text: {skip_special_tokens_transformers(tokenizer, output_text)}")

        >>> clean_text: assistant

            Easy one!

            121 times 10 is 1,210.
    """
    tokens = tokenizer.tokenize(output)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokenizer.decode(token_ids, skip_special_tokens=True)
