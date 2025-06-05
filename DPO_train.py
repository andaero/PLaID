import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

from peft import PeftModel
from llm_finetune import MAX_LENGTH

from llm_finetune import smart_tokenizer_and_embedding_resize


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def main(script_args, training_args, model_args, args):
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    llama_options = args.type.split("-")
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
        if args.online:
            model_string = f"Qwen/{base_str}"
        else:
            model_string = f"qwen-local/{base_str}"
    elif args.online:
        model_string = f"meta-llama/{base_str}"
    else:
        model_string = f"llama/{base_str}"

    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # helps set requires_grad = true
    model.enable_input_require_grads()

    if args.qwen:
        if args.online:
            tokenizer_string = f"Qwen/{base_str}"
        else:
            tokenizer_string = f"qwen-local/{base_str}-tokenizer"
    elif args.online:
        tokenizer_string = f"meta-llama/{base_str}"
    else:
        tokenizer_string = f"llama/{base_str}-tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    if args.max_dir:
        checkpoint_dirs = list(Path(model_args.model_name_or_path).glob("checkpoint-*"))
        checkpoint_nums = [int(d.name.split("-")[1]) for d in checkpoint_dirs]
        most_recent_idx = checkpoint_nums.index(max(checkpoint_nums))
        most_recent_dir = str(checkpoint_dirs[most_recent_idx])
        if args.dpo:
            model_args.model_name_or_path = str(Path(most_recent_dir) / "train_dpo")
            print(f"Using DPO model from {model_args.model_name_or_path}")
        else:
            model_args.model_name_or_path = str(Path(most_recent_dir))

    model = PeftModel.from_pretrained(
        model,
        model_args.model_name_or_path,
        is_trainable=True,
        adapter_name="train_dpo",
        device_map="auto",
    )

    peft_config = get_peft_config(model_args)
    print("peft_config", peft_config)

    if peft_config is None:
        if args.ref_model:
            print(f"Loading reference model from {args.ref_model}")
            model.load_adapter(args.ref_model, adapter_name="reference")
        else:
            print(f"Loading reference model from {model_args.model_name_or_path}")
            model.load_adapter(model_args.model_name_or_path, adapter_name="reference")

    else:
        ref_model = None

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_name)
    # print first 10 rows of dataset
    print(dataset["train"][0])

    ##########
    # Training
    ################
    # ANDY NOTE, CODE WORKS UP TO HERE SO FAR
    # NEED TO MODIFY TRAINING ARG STUFF FOR OUR PREF DATASET NOW
    os.environ["WANDB_PROJECT"] = "PLaID Finetune"
    os.environ["WANDB_LOG_MODEL"] = "end"  # log last model checkpoint
    os.environ["WANDB_MODE"] = "offline"
    if args.online:
        # os.environ["WANDB_CACHE_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_CONFIG_DIR"] = "/mnt/cs/cs152/individual/wandb"
        # os.environ["WANDB_DIR"] = "/mnt/cs/cs152/individual/wandb"
        os.environ["WANDB_DATA_DIR"] = "/mnt/cs/cs152/individual/wandb"
    else:
        os.environ["WANDB_CACHE_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_CONFIG_DIR"] = "/scratch/user/u.ax227774/wandb"
        os.environ["WANDB_DATA_DIR"] = "/scratch/user/u.ax227774/wandb"
    training_args.model_adapter_name = "train_dpo"
    training_args.ref_adapter_name = "reference"
    trainer = DPOTrainer(
        model,
        # ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
        parser.add_argument("--type", type=str, required=True)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--llama2", action="store_true", default=False)
        parser.add_argument("--qwen", action="store_true", default=False)
        parser.add_argument("--online", action="store_true", default=False)
        parser.add_argument("--max_dir", action="store_true", default=False)
        parser.add_argument("--dpo", action="store_true", default=False)
        parser.add_argument("--ref_model", type=str, default=None)

        # for no wandb, use --report_to none
    return parser


if __name__ == "__main__":
    parser = make_parser()
    print(parser.parse_args_and_config())
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    training_args.per_device_train_batch_size = custom_args.batch_size
    training_args.per_device_eval_batch_size = custom_args.batch_size
    # print(f"SCRIPT ARGS: {script_args}")
    # print(f"TRAINING ARGS: {training_args}")
    print(f"MODEL ARGS: {model_args}")
    main(script_args, training_args, model_args, custom_args)
