import torch

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model,TaskType

class LoraCell(torch.nn.Module):
    def __init__(self, base_model, lora_r = 16, lora_alpha = 32, lora_dropout = 0.05, target_modules = None, bias = "none"):
        super().__init__()
        self.model = base_model

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, input_ids, **kwargs):
        out = self.model(input_ids, **kwargs)
        if 'labels' in kwargs:
            labels = kwargs['labels']
        else:
            labels = input_ids

        logits = out.logits
        labels = labels.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        out.loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
            )
    def save_pretrained(self, path):
        """Save LoRA adapter weights"""
        self.model.save_pretrained(path)
        
    def load_pretrained(self, path):
        """Load LoRA adapter weights"""
        self.model = self.model.from_pretrained(path) 



