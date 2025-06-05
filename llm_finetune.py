"""
This file depends on and modifies code from Meta's crystal-text-llm repository, which is MIT-licensed.
The original license is preserved.
"""

import os
import glob
import argparse
import torch
import random
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pathlib import Path

from dataclasses import dataclass
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model

from utils import get_crystal_string_wyckoff_pyx


# Check if CUDA is available and set the device accordingly
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set the GPU device index (0 for the first GPU)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
else:
    print("Using CPU")

IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_crystal_string(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = (
        " ".join(["{0:.1f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + "\n" + " ".join(["{0:.2f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    return crystal_str


class CifDataset(Dataset):
    def __init__(
        self,
        csv_fn,
        format_options={},
        llama_tokenizer=None,
        w_attributes=False,
    ):
        super().__init__()

        if not os.path.exists(csv_fn) and not glob.glob(csv_fn):
            raise ValueError(f"CSV file {csv_fn} does not exist")

        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(csv_fn)])
        self.inputs = df.to_dict(orient="records")

        self.llama_tokenizer = llama_tokenizer

        self.format_options = format_options
        self.w_attributes = w_attributes

    def crystal_string(self, input_dict):
        k = "cif" if "cif" in input_dict else "cif_str"
        # change this to get_crystal_string if no wyckoff
        return get_crystal_string_wyckoff_pyx(input_dict[k])

    def generation_task(self, input_dict):
        prompt = "Below is a description of a bulk material. "

        all_attributes = [
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "spacegroup.number",
        ]

        valid_attributes = list(set(all_attributes) & set(input_dict.keys()))
        # sample a random collection of attributes
        num_attributes = random.randint(0, len(valid_attributes))
        if num_attributes > 0 and self.w_attributes:
            attributes = random.sample(valid_attributes, num_attributes)
            attributes = ["pretty_formula"] + attributes

            prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

            for attr in attributes:
                if attr == "elements":
                    prompt += f"{prompt_lookup[attr]} {', '.join(input_dict[attr])}. "
                elif attr in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
                    if not pd.isna(input_dict[attr]):
                        prompt += f"{prompt_lookup[attr]} {round(float(input_dict[attr]), 4)}. "
                else:
                    prompt += f"{prompt_lookup[attr]} {input_dict[attr]}. "
        # prompt += (
        #     "Generate a crystal structure with the chemical formula, space group,"
        #     "the lengths and angles of the lattice vectors, the number of sites, "
        #     "and the element type and coordinates and wyckoff positions for each atom within the lattice. "
        #     "Enclose the crystal structure within <crystal> </crystal> tags.\n"
        # )
        # old prompt non wyckoff
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )

        crystal_str = self.crystal_string(input_dict)
        tokens = self.llama_tokenizer(
            crystal_str + self.llama_tokenizer.eos_token,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def infill_task(self, input_dict):
        prompt = (
            "Below is a partial description of a bulk material where one "
            'element has been replaced with the string "[MASK]":\n'
        )

        k = "cif" if "cif" in input_dict else "cif_str"
        structure = Structure.from_str(input_dict[k], fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)

        crystal_string = self.crystal_string(input_dict)

        partial_crystal_str = crystal_string.replace(species_to_remove, "[MASK]")

        infill_str = prompt + partial_crystal_str + "\n"

        infill_str += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        infill_str += str(species_to_remove) + self.llama_tokenizer.eos_token

        tokens = self.llama_tokenizer(
            infill_str,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def tokenize(self, input_dict):
        if random.random() < 0.66:
            tokens = self.generation_task(input_dict)
        else:
            tokens = self.infill_task(input_dict)

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.inputs[index]
        vals = self.tokenize(vals)
        return vals


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances)
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_datasets(args, llama_tokenizer, transform_args={}):
    format_options = {
        "permute_composition": args.format_permute_composition,
        "permute_structure": args.format_permute_structure,
    }

    datasets = {
        "train": CifDataset(
            str(args.data_path / "train.csv"),
            format_options,
            llama_tokenizer=llama_tokenizer,
            w_attributes=args.w_attributes,
        ),
        "val": CifDataset(
            str(args.data_path / "val.csv"),
            format_options,
            llama_tokenizer=llama_tokenizer,
            w_attributes=args.w_attributes,
        ),
    }

    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "PLaID Finetune"
    os.environ["WANDB_LOG_MODEL"] = "end"  # log last model checkpoint
    os.environ["WANDB_MODE"] = "offline"

    if args.teapot:
        # os.environ["WANDB_CACHE_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_CONFIG_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_DIR"] = "/mnt/cs/cs152/individual/wandb"
        os.environ["WANDB_DATA_DIR"] = "/mnt/cs/cs152/individual/wandb"
    else:
        os.environ["WANDB_CACHE_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_CONFIG_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_DATA_DIR"] = "/scratch/user/u.ax227774/wandb"

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp4,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        log_level="debug",
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["crystal_ids"],  # this is just to get trainer to behave how I want
    )
    return training_args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def setup_model_online(args):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def qwen_model_string_online(model_size):
        return f"Qwen/Qwen2.5-{model_size.lower()}"

    def llama3_model_string(model_size):
        return f"meta-llama/Llama-3.1-{model_size.lower()}"

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    if args.llama2:
        model_string = llama2_model_string(model_size, is_chat)
    elif args.qwen:
        model_string = qwen_model_string_online(model_size)
    else:
        model_string = llama3_model_string(model_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if args.qwen:
        export_string = model_string.replace("Qwen", "qwen-local", 1)
    else:
        export_string = model_string.replace("meta-llama", "llama")
    # export
    model.save_pretrained(export_string)
    print(f"Exported model to {export_string}")

    llama_tokenizer = AutoTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    export_tokenizer_string = export_string + "-tokenizer"
    # export
    llama_tokenizer.save_pretrained(export_tokenizer_string)
    print(f"Exported tokenizer to {export_tokenizer_string}")
    return


def setup_model(args, rank):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama3_model_string(model_size, chat):
        return f"meta-llama/Llama-3.1-{model_size.lower()}"

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    if args.llama2:
        model_string = llama2_model_string(model_size, is_chat)
    else:
        model_string = llama3_model_string(model_size, is_chat)

    quantization_config = BitsAndBytesConfig(load_in_4bit=args.fp4)
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    llama_tokenizer = AutoTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer


def setup_model_offline(args, rank):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def qwen_model_string_online(model_size):
        return f"Qwen2.5-{model_size.lower()}"

    def llama3_model_string_online(model_size):
        return f"Llama-3.1-{model_size.lower()}"

    def llama2_model_string_online(model_size, chat):
        chat = "chat-" if chat else ""
        return f"Llama-2-{model_size.lower()}-{chat}hf"

    if args.llama2:
        base_str = llama2_model_string_online(model_size, is_chat)
    elif args.qwen:
        base_str = qwen_model_string_online(model_size)
    else:
        base_str = llama3_model_string_online(model_size)

    if args.qwen:
        if args.teapot:
            model_string = f"Qwen/{base_str}"
        else:
            model_string = f"qwen-local/{base_str}"
    elif args.teapot:
        model_string = f"meta-llama/{base_str}"
    else:
        model_string = f"llama/{base_str}"

    quantization_config = BitsAndBytesConfig(load_in_4bit=args.fp4)
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    if args.qwen:
        if args.teapot:
            tokenizer_string = f"Qwen/{base_str}"
        else:
            tokenizer_string = f"qwen-local/{base_str}-tokenizer"
    elif args.teapot:
        tokenizer_string = f"meta-llama/{base_str}"
    else:
        tokenizer_string = f"llama/{base_str}-tokenizer"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer


def setup_trainer(args):
    training_args = setup_training_args(args)

    if args.online:
        setup_model_online(args)
        return
    else:
        model, llama_tokenizer = setup_model_offline(args, training_args.local_rank)

    datasets = setup_datasets(args, llama_tokenizer)
    # calculate the avg num of tokens per sample in the train set
    train_dataset = datasets["train"]
    train_dataset_lens = [len(x["input_ids"]) for x in train_dataset]
    print(train_dataset_lens)
    print(len(train_dataset_lens))
    avg_train_len = sum(train_dataset_lens) / len(train_dataset_lens)
    print(f"Average number of tokens per sample in the train set: {avg_train_len}")

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        # find max dir
        if args.max_dir:
            checkpoint_dirs = list(Path(args.resume_dir).glob("checkpoint-*"))
            checkpoint_nums = [int(d.name.split("-")[1]) for d in checkpoint_dirs]
            most_recent_idx = checkpoint_nums.index(max(checkpoint_nums))
            most_recent_dir = str(checkpoint_dirs[most_recent_idx])
            args.resume_dir = most_recent_dir
        # resume from dir
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        print("Starting training")
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="exp")
    parser.add_argument("--model_name", default="7b")
    parser.add_argument("--qwen", action="store_true", default=False)
    parser.add_argument("--fp4", action="store_true", default=False)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/basic")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=2000, type=int)
    parser.add_argument("--save-freq", default=1000, type=int)
    parser.add_argument(
        "--format-permute-composition", action="store_true", default=False
    )
    parser.add_argument(
        "--format-permute-structure", action="store_true", default=False
    )
    parser.add_argument("--w-attributes", type=int, default=1)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--max-dir", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--online", action="store_true", default=False)

    parser.add_argument("--llama2", action="store_true", default=False)
    parser.add_argument("--teapot", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
