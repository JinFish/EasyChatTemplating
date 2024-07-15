# EasyChatTemplating

Since **vllm** does not automatically load chat templates, it necessitates searching for the corresponding template and then pasting it, which can be quite cumbersome. 

In contrast, Hugging Face offers a method for utilizing chat templates. Therefore, this repository invokes Hugging Face's approach to convert user input **prompts** into **conversation format** through chat templates before feeding them into the large model. Additionally, it processes the output from the large model, removing special tokens to ensure the result is clean.

## Setup

```bash
git clone git@github.com:JinFish/EasyChatTemplating.git
cd EasyChatTemplating
pip install -e .
```

## Example
See `EasyChatTemplating/EasyChatTemplating/examples.py`, or you run the following code:
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from EasyChatTemplating.util_tools import convert_userprompt_transformers, skip_special_tokens_transformers

# Your tokenizer path
tokenizer = AutoTokenizer.from_pretrained('../../pretrained_models/llama3-chat')
# Your model path
llm = LLM(model="../../pretrained_models/llama3-chat")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
prompt = "What is 100 times 101?"
message = convert_userprompt_transformers(tokenizer, prompt, add_generation_prompt=True)
print("prompt:", prompt)
print("message:", message)
output = llm.generate(message, sampling_params)
output_text = output[0].outputs[0].text
print(f"output_text: {output_text}")
print(f"clean_text: {skip_special_tokens_transformers(tokenizer, output_text)}")


>>> prompt: What is 100 times 101?
message: <|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is 100 times 101?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# set add_generation_prompt=True
output_text: 100 × 101 = 10100
clean_text: 100 × 101 = 10100
# set add_generation_prompt=False
output_text: <|start_header_id|>assistant<|end_header_id|>

100 x 101 = 10100
clean_text: assistant

100 x 101 = 10100
```