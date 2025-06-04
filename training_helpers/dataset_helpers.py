import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer
import hashlib

def load_tokenizer(model_name: str, cfg: dict):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=cfg.get("hub_token"),
        trust_remote_code=True,
        padding_side="left", 
        truncation_side="right"
    )
    if tok.pad_token_id is None:      # e.g. Llama‑3, Qwen‑2 FlashAttn
        tok.pad_token = tok.eos_token
    tok.add_eos_token = True
    tok.truncation = True
    return tok


def load_sft_datasets(cfg: dict):
    """
    Return (train_ds, eval_ds)
    If cfg["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=cfg["datasets"][0]["path"],
        split="train",          # guarantees Dataset, not DatasetDict
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    def combine_prompt(example):
        # Handles the case when "input" (the context) may be empty
        if example["input"]:
            prompt = f"{example['prompt']}\n{example['input']}"
        else:
            prompt = example["prompt"]
        example["prompt"] = prompt
        return example                  
    
    if cfg["datasets"][0]["field_input"] is not None:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            cfg["datasets"][0]["field_instruction"]:   "prompt",
            cfg["datasets"][0]["field_input"]:   "input",
            cfg["datasets"][0]["field_output"]:   "completion",
        })
        ds_train = ds_train.map(combine_prompt)
    else:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            cfg["datasets"][0]["field_instruction"]:   "prompt",
            cfg["datasets"][0]["field_output"]:   "completion",
        })

    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                + ex.get("completion", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )
    
    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None
    
    return train_ds, eval_ds


def load_dpo_datasets(cfg: dict):
    """
    Return (train_ds, eval_ds) ready for TRL‑DPO.
    If cfg["val_set_size"] is 0 → eval_ds is None.
    """
    # Load dataset (guarantees a Dataset, not a DatasetDict)
    ds_train = load_dataset(
        "json",
        data_files=cfg["datasets"][0]["path"],
        split="train",
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    # Standardise column names
    ds_train = ds_train.rename_columns({
        cfg["datasets"][0]["field_prompt"]:   "prompt",
        cfg["datasets"][0]["field_chosen"]:   "chosen",
        cfg["datasets"][0]["field_rejected"]: "rejected",
    })
    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                + ex.get("chosen", "")        # use "rejected" if that’s your column name
                + ex.get("rejected", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )

    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds


def load_grpo_datasets(cfg: dict):
    """
    Return (train_ds, eval_ds) ready for TRL‑GRPO.
    If cfg["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=cfg["datasets"][0]["path"],
        split="train",          # guarantees Dataset, not DatasetDict
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    # Standardise column names
    ds_train = ds_train.rename_columns({
        cfg["datasets"][0]["field_prompt"]:   "prompt",
    })

    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )


    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds
