import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer

# ─── Cleanlab helpers ──────────────────────────────────────────────────────────
from cleanlab.outlier import OutOfDistribution      # ⓒ cleanlab 2.x
from sentence_transformers import SentenceTransformer
import numpy as np, torch, math


def _clean_dataset_with_cleanlab(ds, cfg, cols):
    """
    Returns a *filtered* copy of `ds` containing only the most in-distribution
    rows according to Cleanlab's OOD detector.

    Args:
        ds   : HuggingFace `datasets.Dataset`
        cfg  : your existing config dict
        cols : tuple[str]   (# columns to concat into one string per row)

    Config flags (with sensible fallbacks):
        cfg["cleanlab"]           → bool      | default False
        cfg["cleanlab_keep_frac"] → float [0,1] | default 0.90
        cfg["embed_model"]        → str       | default all-MiniLM-L6-v2
        cfg["embed_batch"]        → int       | default 64
    """
    if not cfg.get("cleanlab", False):
        return ds   # no-op ─ user disabled

    keep_frac = float(cfg.get("cleanlab_keep_frac", 0.90))
    assert 0 < keep_frac < 1, "keep_frac must be inside (0,1)"

    # 1 · Flatten each row into a single free-text string
    concat = ["\n".join(str(ex[c]).strip() for c in cols if ex.get(c))
              for ex in ds]

    # 2 · Compute sentence embeddings (≈1.2 GB/min on A100)
    model_name = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    sbert = SentenceTransformer(model_name,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    emb = sbert.encode(concat,
                       batch_size=int(cfg.get("embed_batch", 64)),
                       show_progress_bar=False,
                       convert_to_numpy=True,
                       normalize_embeddings=True)

    # 3 · Score each row with Cleanlab
    ood = OutOfDistribution()
    scores = ood.fit_score(features=emb)   # lower = more in-distribution

    # 4 · Keep the most in-distribution rows
    thresh = np.quantile(scores, keep_frac)      # e.g. 90 % kept
    idx_keep = np.where(scores <= thresh)[0].tolist()
    return ds.select(idx_keep)



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

     # Cleanlab pass (drops the noisiest/outlier rows)
    ds_train = _clean_dataset_with_cleanlab(
        ds_train,
        cfg,
        cols=("prompt", "completion")  # columns to concat
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

    # Cleanlab pass (drops the noisiest/outlier rows)
    ds_train = _clean_dataset_with_cleanlab(
        ds_train,
        cfg,
        cols=("prompt", "chosen", "rejected")  # columns to concat
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

    # Cleanlab pass (drops the noisiest/outlier rows)
    ds_train = _clean_dataset_with_cleanlab(
        ds_train,
        cfg,
        cols=("prompt")  # columns to concat
    )

    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds
