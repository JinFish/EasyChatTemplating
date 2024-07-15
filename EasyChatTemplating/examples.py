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
