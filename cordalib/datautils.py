import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data

'''
llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""
'''

llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""










SYSTEM_PROMPT_NEUTRAL = (
    "You are a helpful, concise assistant. "
    "Follow the user's instructions carefully and answer in clear Markdown."
)



SYSTEM_PROMPT_SAFETY = (
    "You are a helpful and responsible assistant. Be accurate and concise. "
    "If a request is unsafe or illegal, refuse and briefly explain why, "
    "then offer a safer alternative if possible."
)



def pack_for_chat(tokenizer, user_text: str, system_prompt: str | None = None) -> str:

    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        
        msgs = [{"role": "user", "content": user_text}] 
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
          
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
       
    # fallback: Llama-style
    if system_prompt:
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_text} [/INST]"
    return f"[INST] {user_text} [/INST]"


















def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
        #f = open(f)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
   
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
   
    import random
    random.seed(seed)
    if not os.path.exists("cache"):
        os.makedirs("cache")

    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
      
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name=="ptb":
        traindata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "traivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train")
        tot_text = "\n\n".join(traindata["question"])        
    elif name == "alpaca":
        # this is for chat models
        data_path="data/alpaca_data.json"
        list_data_dict = jload(data_path) 
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        
        #random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        #selected_data_dict = [list_data_dict[i] for i in random_indices]
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["instruction"], response=example["output"]) 
                trainenc=tokenizer(s, return_tensors="pt") 
                inp=trainenc.input_ids[:, :seqlen]
              
                attention_mask = torch.ones_like(inp) 
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "MetaMATH":
        data_path="data/MetaMathQA-395K.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["response"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "codefeedback":
        data_path="data/CodeFeedback-Filtered-Instruction.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        #selected_data_dict=random.sample(list_data_dict, nsamples)
        random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        selected_data_dict = [list_data_dict[i] for i in random_indices]        
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["answer"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s) 
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "WizLMinstruct":
        data_path="data/WizardLM_evol_instruct_V2_143k.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list)) 
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item) 
            list_data_dict.append(dict_item) 
            assert isinstance(dict_item, dict) 
        #list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["conversation"][0]["human"], response=example["conversation"][0]["assistant"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen] 
                attention_mask = torch.ones_like(inp) 
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset      
    
    
    elif name == "AdvBench":
        try:
            traindata = load_dataset("./data/AdvBench", split="train") 
        except Exception:
            traindata = load_dataset("walled-ai/advbench", split="train")
        cols = traindata.column_names
        if "prompt" in cols:
            prompts = traindata["prompt"]
        elif "question" in cols:
            prompts = traindata["question"]
        elif "instruction" in cols:
            prompts = traindata["instruction"]
        else:
          
            prompts = [str(x) for x in traindata[cols[0]]]

     
        prompts = prompts[:520]

       
        import random
        random.seed(seed)
        if nsamples is not None and nsamples < len(prompts):
            selected = random.sample(prompts, nsamples)
        else:
            selected = prompts

        traindataset = []
        for s in selected:
            enc = tokenizer(str(s), return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)   
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

     
        if len(selected) > 0:
            print("example advbench prompt:", selected[0])

        # torch.save(traindataset, cache_file)
        return traindataset
    
      
    elif name.startswith("ORBench") or name.startswith("or-bench"):
    
        if ":" in name:
            subset = name.split(":", 1)[1].strip()   # e.g. "ORBench:or-bench-toxic"
        else:
            subset = name if name.startswith("or-bench") else "or-bench-toxic"

       
        try:
          
            ds = load_dataset("./data/bench-llm/or-bench", "or-bench-toxic", split="train")
        except Exception:
            ds = load_dataset("bench-llm/or-bench", subset, split="train")

        from collections import defaultdict, Counter
        rnd = random.Random(seed)


        buckets = defaultdict(list)
        for i, cat in enumerate(ds["category"]):
            buckets[str(cat)].append(i)

        labels = sorted(buckets.keys()) 
        if len(labels) == 0:
            raise ValueError(f"or-bench subset '{subset}' has no 'category' column.")

       
        if nsamples is None:
            chosen_idx = [i for lab in labels for i in buckets[lab]]
        else:
            k = len(labels)
            base, rem = divmod(nsamples, k)

            chosen_idx = []
            for j, lab in enumerate(labels):
                idxs = buckets[lab][:]
                rnd.shuffle(idxs) 
                need = base + (1 if j < rem else 0)
                take = min(need, len(idxs))
                chosen_idx.extend(idxs[:take])

            
            if len(chosen_idx) < nsamples:
                used = set(chosen_idx)
                pool = [i for lab in labels for i in buckets[lab] if i not in used]
                rnd.shuffle(pool)
                chosen_idx.extend(pool[: (nsamples - len(chosen_idx))])

            chosen_idx = chosen_idx[:nsamples]

 
        selected_prompts = [str(ds["prompt"][i]) for i in chosen_idx]

        traindataset = []
        for s in selected_prompts:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if selected_prompts:
            print("example or-bench prompt:", selected_prompts[0])

       
        try:
            dist = Counter([str(ds["category"][i]) for i in chosen_idx])
            print("OR-Bench category counts:", dict(dist))
        except Exception:
            pass

        torch.save(traindataset, cache_file)
        return traindataset

    elif name.startswith("sorry-bench"):
        """
        Use only samples whose prompt_style is in {"base", "role_play"}.
        Randomize independently within each class (using a seed derived from the global seed and the class name),
        distribute nsamples evenly across the two classes, and if a class is short, top up from the combined pool of the two classes. Finally, 
        wrap with the chat template and encode.
        """
        import hashlib
        from collections import Counter

        try:
    
            ds = load_dataset(
                "./data/sorry-bench/sorry-bench-202406",
                split="train"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sorry-bench: {e}")

        def extract_user_turn(d, i):
            t = d[i]["turns"]
            return t[0] if isinstance(t, (list, tuple)) and len(t) > 0 else str(t)

        target_styles = ["base", "role_play"]

        style2idxs = {st: [] for st in target_styles}
        for i, st in enumerate(ds["prompt_style"]):
            st = str(st)
            if st in style2idxs:
                style2idxs[st].append(i)

        available = [st for st in target_styles if len(style2idxs[st]) > 0]
        if not available:
            raise RuntimeError("No samples for 'base' or 'role_play' in sorry-bench-202406.")

        per_cls_base, rem = divmod(nsamples, len(available))
        need_per_style = {
            st: per_cls_base + (1 if idx < rem else 0)
            for idx, st in enumerate(available)
        }

        def derived_seed(base_seed: int, tag: str) -> int:
            h = hashlib.sha256(f"{base_seed}-{tag}".encode()).digest()
            return int.from_bytes(h[:8], "big")

        chosen_idx = []
        for st in available:
            idxs = style2idxs[st][:]  
            rnd_cls = random.Random(derived_seed(seed, f"sorry-{st}"))
            rnd_cls.shuffle(idxs)    
            need = need_per_style[st]
            take = min(need, len(idxs))
            chosen_idx.extend(idxs[:take])

       
        if len(chosen_idx) < nsamples:
            used = set(chosen_idx)
            pool = [i for st in available for i in style2idxs[st] if i not in used]
            rnd_pool = random.Random(derived_seed(seed, "sorry-pool"))
            rnd_pool.shuffle(pool)
            gap = nsamples - len(chosen_idx)
            chosen_idx.extend(pool[:max(0, gap)])

       
        chosen_idx = chosen_idx[:nsamples]

   
        traindataset = []
        for i in chosen_idx:
            user_prompt = extract_user_turn(ds, i)
            enc = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp) 
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if chosen_idx:
            print("example sorry-bench (base/role_play) prompt:", extract_user_turn(ds, chosen_idx[0]))
            try:
                dist = Counter([str(ds["prompt_style"][i]) for i in chosen_idx])
                print("sorry-bench counts (base/role_play):", dict(dist))
            except Exception:
                pass

        torch.save(traindataset, cache_file)
        return traindataset


   
    

    elif name in ("RealToxicityPrompts", "real-toxicity-prompts", "allenai/real-toxicity-prompts"):
     
        ds = load_dataset("./data/allenai/real-toxicity-prompts", split="train")
        ds_hi = ds.filter(
            lambda r: isinstance(r.get("prompt"), dict)
            and r["prompt"].get("toxicity") is not None
            and float(r["prompt"]["toxicity"]) > 0.5
        )
        prompts = [
            str(r["prompt"].get("text", "")).strip()
            for r in ds_hi
            if r.get("prompt") and r["prompt"].get("text")
        ]
           
        if not prompts:
            raise ValueError("No prompts with toxicity > 0.5 were found in allenai/real-toxicity-prompts.")

        import random
        random.seed(seed)
        if nsamples is not None and nsamples < len(prompts):
            selected = random.sample(prompts, nsamples)
        else:
            selected = prompts if nsamples is None else prompts[:nsamples]

       
        traindataset = []
        for s in selected:
            enc = tokenizer(str(s), return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if selected:
            print("example real-toxicity prompt:", selected[0])


        return traindataset


    elif name.startswith("MixData"):
      
      
        if ":" in name:
            jsonl_path = name.split(":", 1)[1].strip()
        
        else:
           
            jsonl_path = "./Evaluate/MixData_1500.jsonl"  

  
        ds = load_dataset("json", data_files=jsonl_path, split="train")

        if "prompt" not in ds.column_names:
            raise ValueError(f"No 'prompt' field found in {jsonl_path}. Available columns: {ds.column_names}")

        prompts = [str(p).strip() for p in ds["prompt"]]
        prompts = [p for p in prompts if p]

        rnd = random.Random(seed)
        if nsamples is None or nsamples >= len(prompts):
            selected = prompts
        else:
            selected = rnd.sample(prompts, nsamples)

        traindataset = []
        for s in selected:
     
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attn = torch.ones_like(inp)  # 未 pad，直接全 1
            traindataset.append({"input_ids": inp, "attention_mask": attn})

        if selected:
            print("example mixed-jsonl prompt:", selected[0])

    
        return traindataset


    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset



def get_null_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    import random
    random.seed(seed)
    if not os.path.exists("cache"):
        os.makedirs("cache")
 
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name=="ptb":
        traindata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "traivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train")
        tot_text = "\n\n".join(traindata["question"])        
    elif name == "alpaca":
        # this is for chat models
        data_path="data/alpaca_data.json"
        list_data_dict = jload(data_path) 
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "": 
                s=llama_chat_format.format(instruction=example["instruction"], response=example["output"]) 
                trainenc=tokenizer(s, return_tensors="pt") 
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp) 
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "MetaMATH":
        data_path="data/MetaMathQA-395K.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["response"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "codefeedback":
        data_path="data/CodeFeedback-Filtered-Instruction.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        #selected_data_dict=random.sample(list_data_dict, nsamples)
        random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        selected_data_dict = [list_data_dict[i] for i in random_indices]        
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["answer"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s) 
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "WizLMinstruct":
        data_path="data/WizardLM_evol_instruct_V2_143k.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list)) 
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item) 
            list_data_dict.append(dict_item) 
            assert isinstance(dict_item, dict) 
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples) 
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["conversation"][0]["human"], response=example["conversation"][0]["assistant"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen] 
                attention_mask = torch.ones_like(inp) 
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset      
    
       
    elif name == "AdvBench":
        try:
            traindata = load_dataset("./data/AdvBench", split="train") 
        except Exception:
            traindata = load_dataset("walled-ai/advbench", split="train")


        cols = traindata.column_names
        if "prompt" in cols:
            prompts = traindata["prompt"]
        elif "question" in cols:
            prompts = traindata["question"]
        elif "instruction" in cols:
            prompts = traindata["instruction"]
        else:
            prompts = [str(x) for x in traindata[cols[0]]]

        prompts = prompts[:520]
        import random
        random.seed(seed)
        if nsamples is not None and nsamples < len(prompts):
            selected = random.sample(prompts, nsamples)
        else:
            selected = prompts

        traindataset = []
        for s in selected:
            enc = tokenizer(str(s), return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)   
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        if len(selected) > 0:
            print("example advbench prompt:", selected[0])

        return traindataset
    
    elif name.startswith("ORBench") or name.startswith("or-bench"):
        if ":" in name:
            subset = name.split(":", 1)[1].strip()   
        else:
            subset = name if name.startswith("or-bench") else "or-bench-toxic"

        
        try:
            ds = load_dataset("./data/bench-llm/or-bench", "or-bench-toxic", split="train")
        except Exception:
            ds = load_dataset("bench-llm/or-bench", subset, split="train")

        from collections import defaultdict, Counter
        rnd = random.Random(seed)

        buckets = defaultdict(list)
        for i, cat in enumerate(ds["category"]):
            buckets[str(cat)].append(i)

        labels = sorted(buckets.keys())  
        if len(labels) == 0:
            raise ValueError(f"or-bench subset '{subset}' has no 'category' column.")
        if nsamples is None:
            chosen_idx = [i for lab in labels for i in buckets[lab]]
        else:
            k = len(labels)
            base, rem = divmod(nsamples, k)

            chosen_idx = []
            for j, lab in enumerate(labels):
                idxs = buckets[lab][:]
                rnd.shuffle(idxs)  
                need = base + (1 if j < rem else 0)
                take = min(need, len(idxs))
                chosen_idx.extend(idxs[:take])

            if len(chosen_idx) < nsamples:
                used = set(chosen_idx)
                pool = [i for lab in labels for i in buckets[lab] if i not in used]
                rnd.shuffle(pool)
                chosen_idx.extend(pool[: (nsamples - len(chosen_idx))])

            chosen_idx = chosen_idx[:nsamples]

        selected_prompts = [str(ds["prompt"][i]) for i in chosen_idx]

        traindataset = []
        for s in selected_prompts:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if selected_prompts:
            print("example or-bench prompt:", selected_prompts[0])

        try:
            dist = Counter([str(ds["category"][i]) for i in chosen_idx])
            print("OR-Bench category counts:", dict(dist))
        except Exception:
            pass

        torch.save(traindataset, cache_file)
        return traindataset

  
    elif name.startswith("sorry-bench"):
        """
        Use only samples whose prompt_style is in {"base", "role_play"}.
        Randomize independently within each class (using a seed derived from the global seed and the class name),
        distribute nsamples evenly across the two classes, and if a class is short, top up from the combined pool of the two classes. Finally, 
        wrap with the chat template and encode.
        """
        import hashlib
        from collections import Counter

        try:
           
            ds = load_dataset(
                "./data/sorry-bench/sorry-bench-202406",
                split="train"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sorry-bench: {e}")

   
        def extract_user_turn(d, i):
            t = d[i]["turns"]
            return t[0] if isinstance(t, (list, tuple)) and len(t) > 0 else str(t)

     
        target_styles = ["base", "role_play"]

        
        style2idxs = {st: [] for st in target_styles}
        for i, st in enumerate(ds["prompt_style"]):
            st = str(st)
            if st in style2idxs:
                style2idxs[st].append(i)

       
        available = [st for st in target_styles if len(style2idxs[st]) > 0]
        if not available:
            raise RuntimeError("No samples for 'base' or 'role_play' in sorry-bench-202406.")

        
        per_cls_base, rem = divmod(nsamples, len(available))
        need_per_style = {
            st: per_cls_base + (1 if idx < rem else 0)
            for idx, st in enumerate(available)
        }

      
        def derived_seed(base_seed: int, tag: str) -> int:
            h = hashlib.sha256(f"{base_seed}-{tag}".encode()).digest()
            return int.from_bytes(h[:8], "big")

      
        chosen_idx = []
        for st in available:
            idxs = style2idxs[st][:] 
            rnd_cls = random.Random(derived_seed(seed, f"sorry-{st}"))
            rnd_cls.shuffle(idxs)    
            need = need_per_style[st]
            take = min(need, len(idxs))
            chosen_idx.extend(idxs[:take])

    
        if len(chosen_idx) < nsamples:
            used = set(chosen_idx)
            pool = [i for st in available for i in style2idxs[st] if i not in used]
            rnd_pool = random.Random(derived_seed(seed, "sorry-pool"))
            rnd_pool.shuffle(pool)
            gap = nsamples - len(chosen_idx)
            chosen_idx.extend(pool[:max(0, gap)])

  
        chosen_idx = chosen_idx[:nsamples]

  
        traindataset = []
        for i in chosen_idx:
            user_prompt = extract_user_turn(ds, i)
            enc = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if chosen_idx:
            print("example sorry-bench (base/role_play) prompt:", extract_user_turn(ds, chosen_idx[0]))
            try:
                dist = Counter([str(ds["prompt_style"][i]) for i in chosen_idx])
                print("sorry-bench counts (base/role_play):", dict(dist))
            except Exception:
                pass

        torch.save(traindataset, cache_file)
        return traindataset


    

    elif name in ("RealToxicityPrompts", "real-toxicity-prompts", "allenai/real-toxicity-prompts"):
        ds = load_dataset("./data/allenai/real-toxicity-prompts", split="train")
        ds_hi = ds.filter(
            lambda r: isinstance(r.get("prompt"), dict)
            and r["prompt"].get("toxicity") is not None
            and float(r["prompt"]["toxicity"]) > 0.5
        )
        prompts = [
            str(r["prompt"].get("text", "")).strip()
            for r in ds_hi
            if r.get("prompt") and r["prompt"].get("text")
        ]
        
        if not prompts:
            raise ValueError("No prompts with toxicity > 0.5 were found in allenai/real-toxicity-prompts.")

      
        import random
        random.seed(seed)
        if nsamples is not None and nsamples < len(prompts):
            selected = random.sample(prompts, nsamples)
        else:
            selected = prompts if nsamples is None else prompts[:nsamples]

       
        traindataset = []
        for s in selected:
            enc = tokenizer(str(s), return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

        if selected:
            print("example real-toxicity prompt:", selected[0])

        return traindataset


    elif name.startswith("MixData"):
    
        if ":" in name:
            jsonl_path = name.split(":", 1)[1].strip()
     
        else:
           
            jsonl_path = "./Evaluate/MixData_1500.jsonl"   

        ds = load_dataset("json", data_files=jsonl_path, split="train")

        if "prompt" not in ds.column_names:
            raise ValueError(f"No 'prompt' field found in {jsonl_path}. Available columns: {ds.column_names}")



        prompts = [str(p).strip() for p in ds["prompt"]]
        prompts = [p for p in prompts if p]

      
        rnd = random.Random(seed)
        if nsamples is None or nsamples >= len(prompts):
            selected = prompts
        else:
            selected = rnd.sample(prompts, nsamples)

      
        traindataset = []
        for s in selected:
           
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=seqlen)
            inp = enc.input_ids[:, :seqlen]
            attn = torch.ones_like(inp)  
            traindataset.append({"input_ids": inp, "attention_mask": attn})

        if selected:
            print("example mixed-jsonl prompt:", selected[0])

       
        return traindataset


    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset



def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
  
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc        
    raise NotImplementedError
