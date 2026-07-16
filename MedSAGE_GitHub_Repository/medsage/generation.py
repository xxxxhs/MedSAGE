from dataclasses import dataclass
import torch

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    do_sample: bool = True

class LocalHFGenerator:
    def __init__(self, model_path, tokenizer_path=None, device_map='auto', torch_dtype='auto'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(torch_dtype, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
        self.model.eval()
    def generate(self, messages, config):
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = '\n'.join([m['content'] for m in messages])
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=config.max_new_tokens, temperature=config.temperature, top_p=config.top_p, do_sample=config.do_sample, pad_token_id=self.tokenizer.eos_token_id)
        gen_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
