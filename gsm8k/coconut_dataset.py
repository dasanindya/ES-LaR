# Reference: https://github.com/facebookresearch/coconut/blob/main/dataset.py
# Modified to support gsm8k_train.json from ES-LaR 
#
# Expected JSON record format (raw GSM8K – steps not yet separated):
#   { "question": "...", "answer": "Step 1\nStep 2\n#### 72" }
#
# Changes vs. original coconut/dataset.py:
#   1. _normalize_record()  – parses the raw GSM8K answer string into the
#                             canonical { question, steps, answer } dict that
#                             the rest of the pipeline expects.
#   2. get_dataset()        – calls _normalize_record() before tokenization;
#                             all other logic (distributed map, assertion) is
#                             identical to the original.
#   3. MyCollator, get_question_latent_dataset, get_cot_latent_dataset –
#                             completely unchanged.

import json
import re
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


# ---------------------------------------------------------------------------
# GSM8K / ES-LaR normalisation helpers
# ---------------------------------------------------------------------------

# Matches the "#### <number>" answer delimiter used in raw GSM8K data.
_ANSWER_DELIMITER = re.compile(r"####\s*(.+)$", re.MULTILINE)

# Strips inline calculator annotations like <<7+13=20>> that GSM8K embeds
# inside reasoning steps. The language model doesn't need to reproduce them.
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def _normalize_record(record: dict, idx: int) -> dict:
    """
    Convert a raw GSM8K record into the canonical CoCoNuT format:

        { "question": str, "steps": list[str], "answer": str, "idx": int }

    Input format expected:
        { "question": "...", "answer": "Step 1\nStep 2\n#### 72" }

    The answer field is split on "####":
      - Everything before it becomes the steps list (one entry per non-empty
        line, with inline calculator annotations like <<7+13=20>> stripped).
      - The number after "####" becomes the answer string.

    Raises ValueError if either required field is missing or "####" is absent.
    """
    if "question" not in record:
        raise ValueError(f"Record {idx} is missing required field 'question'.")
    if "answer" not in record:
        raise ValueError(f"Record {idx} is missing required field 'answer'.")

    question = record["question"].strip()
    raw_answer = record["answer"].strip()

    match = _ANSWER_DELIMITER.search(raw_answer)
    if not match:
        raise ValueError(
            f"Record {idx}: could not find '####' delimiter in answer field. "
            f"Got: {raw_answer!r}"
        )

    answer = match.group(1).strip()
    reasoning_block = raw_answer[: match.start()]
    steps = [
        _CALC_ANNOTATION.sub("", line).strip()
        for line in reasoning_block.splitlines()
        if line.strip()
    ]

    return {
        "question": question,
        "steps": steps,
        "answer": answer,
        "idx": idx,
    }


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------

def get_dataset(path, tokenizer, max_size=1_000_000_000):
    """
    Load *path* (a JSON file), normalise each record with _normalize_record(),
    tokenise, and return a HuggingFace Dataset ready for CoCoNuT training.

    The only change from the original implementation is the insertion of the
    _normalize_record() call to parse the raw GSM8K answer format into the
    separate steps and answer fields that CoCoNuT expects.
    """

    def tokenize_sample(sample):
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }

    # --- Load & normalise ---------------------------------------------------
    raw_data = json.load(open(path))[:max_size]
    data = [_normalize_record(record, idx) for idx, record in enumerate(raw_data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    # --- Tokenise (distributed-aware) ---------------------------------------
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample,
                    remove_columns=list(dataset.features),
                    num_proc=32,
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = dataset.map(
            tokenize_sample,
            remove_columns=list(dataset.features),
            num_proc=32,
        )

    # --- Sanity check -------------------------------------------------------
    d = data[0]
    complete = (
        d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    )
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert complete_tokenized == (
        dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    ), (
        "Tokenisation mismatch on record 0.  Check that _normalize_record() is "
        "producing clean, whitespace-consistent text for this dataset."
    )

    return dataset


# ---------------------------------------------------------------------------
# MyCollator 
# ---------------------------------------------------------------------------

@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.

        E.g.,
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx

        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)

            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature[
                        "input_ids"
                    ].index(self.latent_id)
                else:
                    n_tok_pad = 0

                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = (
                    [self.tokenizer.pad_token_id] * n_tok_pad
                    + feature["input_ids"]
                )
                if "labels" in feature:
                    feature["labels"] = (
                        [self.label_pad_token_id] * n_tok_pad
                        + feature["labels"]
                    )
                feature["attention_mask"] = (
                    [0] * n_tok_pad + feature["attention_mask"]
                )

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"
        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [
                label
                + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


# ---------------------------------------------------------------------------
# get_question_latent_dataset  
# ---------------------------------------------------------------------------

def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    def process_dataset(sample):
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )
        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset,
        remove_columns=list(base_dataset_valid.features),
        num_proc=32,
    )


# ---------------------------------------------------------------------------
# get_cot_latent_dataset  
# ---------------------------------------------------------------------------

def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        if random.random() < configs.uniform_prob:
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all steps
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(
                    sample["steps_tokenized"][n_skip_steps:]
                )
            )
            + sample["answer_tokenized"]
        )
        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset,
                remove_columns=list(base_dataset.features),
                num_proc=32,
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        processed_dataset = base_dataset.map(
            process_dataset,
            remove_columns=list(base_dataset.features),
            num_proc=32,
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset