"""
This file depends on and modifies code from Meta's crystal-text-llm repository, which is MIT-licensed.
The original license is preserved.
"""

import os

import hydra
import pandas as pd
import torch
import transformers
from flytekit import (
    workflow,
)
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel

from utils import get_crystal_string_wyckoff_pyx

from dataclasses import dataclass
from pathlib import Path


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


class StableCrystalDataset(Dataset):
    """
    Dataset class for stable crystal SFT training

    Loads data from the curated stable crystal dataset and handles:
    1. Text prompt generation (gen_str is already formatted)
    2. Crystal structure representation using existing PLaID format
    3. Tokenization for language model training
    """

    def __init__(self, csv_path, tokenizer, use_wyckoff=True):
        super().__init__()

        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file {csv_path} does not exist")

        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.use_wyckoff = use_wyckoff

        print(f"Loaded {len(self.df)} stable crystal samples from {csv_path}")
        print(
            f"Generation types: {self.df['generation_type'].value_counts().to_dict()}"
        )

        # Validate required columns
        required_cols = ["gen_str", "cif"]
        missing_cols = set(required_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def get_crystal_string(self, cif_str):
        """Get crystal string representation using existing PLaID utilities"""
        if self.use_wyckoff:
            return get_crystal_string_wyckoff_pyx(cif_str, translate=False)
        else:
            # Fallback to simple format if wyckoff fails
            from pymatgen.core.structure import Structure

            structure = Structure.from_str(cif_str, fmt="cif")

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

    def create_training_sample(self, row):
        """
        Create training sample from dataset row

        For stable crystals, we use the gen_str as the prompt and the crystal
        representation as the target. The gen_str already contains the proper
        prompt format from the original generation.
        """
        # Extract prompt and expected crystal string
        gen_str = row["gen_str"]
        cif_str = row["cif"]

        # Get crystal string representation
        try:
            crystal_str = self.get_crystal_string(cif_str)
        except Exception as e:
            print(f"Warning: Failed to process crystal structure: {e}")
            return None

        # For stable crystals from our curated dataset, gen_str already contains
        # the full prompt, so we just need to append the crystal structure
        full_text = gen_str + self.tokenizer.eos_token
        return full_text

    def tokenize_sample(self, text):
        """Tokenize a training sample"""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = (
            tokens.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        row = self.df.iloc[index]

        # Create training sample
        text = self.create_training_sample(row)
        if text is None:
            # Return empty sample if processing failed
            return dict(
                input_ids=torch.tensor([]),
                labels=torch.tensor([]),
                input_ids_lens=0,
                labels_lens=0,
            )

        # Tokenize
        return self.tokenize_sample(text)


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


def setup_datasets(cfg: DictConfig, tokenizer):
    """Setup train and validation datasets"""
    datasets = {}
    data_path = Path(cfg.data_path)
    train_path = data_path / "train" / "train.csv"
    val_path = data_path / "val" / "val.csv"

    if train_path.exists():
        datasets["train"] = StableCrystalDataset(
            str(train_path), tokenizer, use_wyckoff=cfg.use_wyckoff
        )
    else:
        raise ValueError(f"Training dataset not found: {train_path}")

    if val_path.exists():
        datasets["val"] = StableCrystalDataset(
            str(val_path), tokenizer, use_wyckoff=cfg.use_wyckoff
        )
    else:
        print(f"Warning: Validation dataset not found: {val_path}")
        datasets["val"] = None

    return datasets


def setup_training_args(cfg: DictConfig):
    output_dir = Path(cfg.expdir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "PLaID++ Finetune"
    os.environ["WANDB_LOG_MODEL"] = "end"  # log last model checkpoint

    if cfg.teapot:
        # os.environ["WANDB_CACHE_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_CONFIG_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_DATA_DIR"] = "/mnt/cs/cs152/individual/wandb"
        print("WANDB_MODE is teapot")
    else:
        os.environ["WANDB_CACHE_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_CONFIG_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_DATA_DIR"] = "/scratch/user/u.ax227774/wandb"

    if cfg.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not cfg.fp4,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=cfg.num_epochs,
        eval_steps=cfg.eval_freq,
        save_steps=cfg.save_freq,
        logging_steps=10,
        log_level="debug",
        eval_strategy="steps",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=cfg.num_warmup_steps,
        # warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.grad_accum,
        output_dir=output_dir,
        run_name=cfg.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        # label_names=["crystal_ids"],  # this is just to get trainer to behave how I want
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


def setup_model_online(cfg: DictConfig):
    llama_options = cfg.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def qwen_model_string_online(model_size):
        return f"Qwen/Qwen2.5-{model_size.lower()}"

    def llama3_model_string(model_size):
        return f"meta-llama/Llama-3.1-{model_size.lower()}"

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    if cfg.llama2:
        model_string = llama2_model_string(model_size, is_chat)
    elif cfg.qwen:
        model_string = qwen_model_string_online(model_size)
    else:
        model_string = llama3_model_string(model_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if cfg.qwen:
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


def setup_model_offline(cfg: DictConfig, rank):
    llama_options = cfg.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def qwen_model_string_online(model_size):
        return f"Qwen2.5-{model_size.lower()}"

    def llama3_model_string_online(model_size):
        return f"Llama-3.1-{model_size.lower()}"

    def llama2_model_string_online(model_size, chat):
        chat = "chat-" if chat else ""
        return f"Llama-2-{model_size.lower()}-{chat}hf"

    if cfg.llama2:
        base_str = llama2_model_string_online(model_size, is_chat)
    elif cfg.qwen:
        base_str = qwen_model_string_online(model_size)
    else:
        base_str = llama3_model_string_online(model_size)

    if cfg.qwen:
        if cfg.teapot:
            model_string = f"Qwen/{base_str}"
        else:
            model_string = f"qwen-local/{base_str}"
    elif cfg.teapot:
        model_string = f"meta-llama/{base_str}"
    else:
        model_string = f"llama/{base_str}"

    quantization_config = BitsAndBytesConfig(load_in_4bit=cfg.fp4)
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    model.enable_input_require_grads()

    if cfg.qwen:
        if cfg.teapot:
            tokenizer_string = f"Qwen/{base_str}"
        else:
            tokenizer_string = f"qwen-local/{base_str}-tokenizer"
    elif cfg.teapot:
        tokenizer_string = f"meta-llama/{base_str}"
    else:
        tokenizer_string = f"llama/{base_str}-tokenizer"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

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

    if cfg.max_dir:
        # find max dir
        checkpoint_dirs = list(Path(cfg.model_path).glob("checkpoint-*"))
        checkpoint_nums = [int(d.name.split("-")[1]) for d in checkpoint_dirs]
        most_recent_idx = checkpoint_nums.index(max(checkpoint_nums))
        most_recent_dir = str(checkpoint_dirs[most_recent_idx])
        model_path = most_recent_dir
    if cfg.ift:
        model_path = str(Path(model_path) / "train_ift")
        print(f"Using IFT model from {model_path}")
    else:
        model_path = cfg.model_path
    print(f"Loading model from {model_path}")
    model = PeftModel.from_pretrained(
        model,
        model_path,
        is_trainable=True,
        adapter_name="train_ift",
        device_map="auto",
    )

    return model, llama_tokenizer


def setup_trainer(cfg: DictConfig):
    training_args = setup_training_args(cfg)

    if cfg.online:
        setup_model_online(cfg)
        return
    else:
        model, llama_tokenizer = setup_model_offline(cfg, training_args.local_rank)

    datasets = setup_datasets(cfg, llama_tokenizer)

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


def finetune_task(cfg: DictConfig):
    trainer = setup_trainer(cfg)

    resume_dir = cfg.resume_dir
    if resume_dir is not None:
        # find max dir
        checkpoint_dirs = list(Path(resume_dir).glob("checkpoint-*"))
        checkpoint_nums = [int(d.name.split("-")[1]) for d in checkpoint_dirs]
        most_recent_idx = checkpoint_nums.index(max(checkpoint_nums))
        most_recent_dir = str(checkpoint_dirs[most_recent_idx])
        resume_dir = most_recent_dir
        # resume from dir
        train_result = trainer.train(resume_from_checkpoint=resume_dir)
    else:
        print("Starting training")
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


@workflow
def finetune_workflow(cfg: DictConfig):
    finetune_task(cfg=cfg)


@hydra.main(config_path="conf/ift", config_name="config", version_base=None)
def main(cfg: DictConfig):
    finetune_workflow(cfg=cfg)


if __name__ == "__main__":
    main()
