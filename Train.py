import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
import transformers
from transformers import Trainer
# from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel


from transformers import default_data_collator  
from utils.dataset_utils import get_preprocessed_dataset 
from utils.config_utils import generate_dataset_config, update_config  
from configs import train_config  
import random
import numpy as np


# ---- Deterministic setup ----
def set_reproducible(seed: int = 42, tf32: bool = False):
    import os, random, numpy as np, torch, transformers

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Python/NumPy/PyTorch 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash(False)
        sdp_kernel.enable_mem_efficient(False)
        sdp_kernel.enable_math(True)
    except Exception:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)



def set_fast_inference(tf32: bool = True):
    import os, torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash(True)
        sdp_kernel.enable_mem_efficient(True)
        sdp_kernel.enable_math(True)   
    except Exception:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)





IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            num_params = num_params * 2 * num_bytes
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    seed: int = 42
    data_seed: int = 42
    group_by_length: bool = False
    dataloader_num_workers: int = 0

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(['train','test','eval'] or HF slice)"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    lora_r: int = field(default=None, metadata={"help": "LoRA rank; None => CorDA/full FT"})
    corda_mode: bool = field(default=True, metadata={"help": "True for CorDA mode"})

    dataset: str = field(default="gsm8k_dataset", metadata={"help": "Name in DATASET_PREPROC (e.g., gsm8k_dataset)"})
    mode: str = field(default="1k_p_0.1", metadata={"help": "subset/mode string (e.g., 1k_p_0.1)"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest",
                  max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = labels = [tok.input_ids[0] for tok in tokenized_list]
    input_ids_lens = labels_lens = [
        tok.input_ids.ne(tokenizer.pad_token_id).sum().item() for tok in tokenized_list
    ]
    return dict(input_ids=input_ids, labels=labels,
                input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str],
               tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tok, sources_tok = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tok["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tok["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([inst[key] for inst in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=ins)) for ins in examples[query]]
    targets = [f"{out}{tokenizer.eos_token}" for out in examples[response]]
    return preprocess(sources, targets, tokenizer)


def train():
    # parser = transformers.HfArgumentParser(TrainingArguments)
    parser = transformers.HfArgumentParser(TrainingArguments, allow_abbrev=False)

    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    seed = getattr(script_args, "seed", 42)
    set_reproducible(seed=seed, tf32=False)


    if script_args.corda_mode:
        print("Train in CorDA mode")
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     script_args.model_name_or_path, device_map="auto", trust_remote_code=True
        # )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  
        )
        for n, p in model.named_parameters():
            if "ALinear" not in n and "BLinear" not in n and p.requires_grad:
                p.requires_grad = False
    elif script_args.lora_r is not None:
        print("Train in LoRA mode")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, device_map="auto"
        )
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            init_lora_weights=True,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Train in Full Finetuning mode")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    print(model)

    for n, p in model.named_parameters():
        print(n, p.requires_grad)
    trn, allp = get_nb_trainable_parameters(model)
    print(f"trainable params: {trn:,d} || all params: {allp:,d} || trainable%: {100 * trn / allp}")

 

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    update_config((train_config,), **vars(script_args))  

    dataset_config = generate_dataset_config(train_config, vars(script_args))

    train_dataset = get_preprocessed_dataset(
        tokenizer=tokenizer,
        dataset_config=dataset_config,  
        split="train",
    )

    data_collator = default_data_collator
   
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    save_dir = os.path.join(script_args.output_dir, "ft")
    os.makedirs(save_dir, exist_ok=True)  
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    train()
