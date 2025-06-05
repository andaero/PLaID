"""
This file depends on and modifies code from Meta's crystal-text-llm repository, which is MIT-licensed.
The original license is preserved.
"""

from pathlib import Path
import os
import torch
import random
import argparse
import pandas as pd
import numpy as np
from collections import deque
import time  # Add this at the top with other imports

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from llm_finetune import get_crystal_string, MAX_LENGTH
from templating import make_swap_table

from utils import parse_fn_wyckoff
from cond_gen.sample_parse import parse_sg_col, seperate_dfs_by_condition

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]

    structure = Structure(
        lattice=Lattice.from_parameters(*(lengths + angles)),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )

    return structure.to(fmt="cif")


def prepare_model_and_tokenizer(args):
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
        if args.online:
            model_string = f"Qwen/{base_str}"
        else:
            model_string = f"qwen-local/{base_str}"
    elif args.online:
        model_string = f"meta-llama/{base_str}"
    else:
        model_string = f"llama/{base_str}"

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2" if args.use_fa2 else None,
    )

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
    model.eval()

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
        checkpoint_dirs = list(Path(args.model_path).glob("checkpoint-*"))
        checkpoint_nums = [int(d.name.split("-")[1]) for d in checkpoint_dirs]
        most_recent_idx = checkpoint_nums.index(max(checkpoint_nums))
        most_recent_dir = str(checkpoint_dirs[most_recent_idx])
        if args.dpo:
            args.model_path = str(Path(most_recent_dir) / "train_dpo")
            print(f"Using DPO model from {args.model_path}")
        else:
            args.model_path = str(Path(most_recent_dir))
    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")
    return model, tokenizer


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


def process_single_item(args_and_data):
    """Process a single generation string and prompt pair."""
    gen_str, prompt, use_wyckoff = args_and_data
    material_str = gen_str.replace(prompt, "")
    material_str = material_str.replace("<crystal>\n", "").replace("\n</crystal>", "")

    try:
        if use_wyckoff:
            cif_str = parse_fn_wyckoff(material_str)
        else:
            cif_str = parse_fn(material_str)
        # Validate the CIF string
        _ = Structure.from_str(cif_str, fmt="cif")

        return {
            "gen_str": gen_str,
            "cif": cif_str,
            "model_name": None,  # Will be added after processing
        }
    except Exception as e:
        print(e)
        return None


def process_batch_parallel(gen_strs, batch_prompts, args, max_workers=None):
    """
    Process multiple items in parallel using ProcessPoolExecutor.

    Args:
        gen_strs: List of generation strings
        batch_prompts: List of prompts
        args: Arguments object containing wyckoff and model_name
        max_workers: Maximum number of worker processes (defaults to CPU count)
    """
    # Create data tuples for processing
    process_data = [
        (gen_str, prompt, args.wyckoff)
        for gen_str, prompt in zip(gen_strs, batch_prompts)
    ]

    outputs = []
    # Use ProcessPoolExecutor to handle the parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs and get futures
        future_to_data = {
            executor.submit(process_single_item, data): data for data in process_data
        }

        # Process completed futures as they finish
        for future in future_to_data:
            try:
                result = future.result()
                if result is not None:
                    result["model_name"] = args.model_name
                    outputs.append(result)
            except Exception as e:
                print(f"Executor exception: {e}")
                continue

    return outputs


def conditional_process_batch_parallel(
    gen_strs, batch_prompts, batch_indices, args, max_workers=None
):
    """
    Process multiple items in parallel and track failed indices for retry.

    Args:
        gen_strs: List of generation strings
        batch_prompts: List of prompts
        batch_indices: List of original indices in the full dataset
        args: Arguments object containing wyckoff and model_name
        max_workers: Maximum number of worker processes (defaults to CPU count)

    Returns:
        tuple: (successful_outputs, failed_indices)
            - successful_outputs: List of successfully processed items
            - failed_indices: List of indices where processing failed
    """
    # Create data tuples for processing
    process_data = [
        (gen_str, prompt, args.wyckoff)
        for gen_str, prompt in zip(gen_strs, batch_prompts)
    ]

    outputs = []
    failed_indices = []

    # Use ProcessPoolExecutor to handle the parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs and get futures
        future_to_data = {
            executor.submit(process_single_item, data): (data, idx)
            for data, idx in zip(process_data, batch_indices)
        }

        # Process completed futures as they finish
        for future in future_to_data:
            data, original_idx = future_to_data[future]
            try:
                result = future.result()
                if result is not None:
                    result["model_name"] = args.model_name
                    outputs.append(result)
                else:
                    # If result is None, processing failed
                    failed_indices.append(original_idx)
            except Exception as e:
                print(f"Executor exception for index {original_idx}: {e}")
                failed_indices.append(original_idx)
                continue

    return outputs, failed_indices


def unconditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    prompts = []
    if args.conditions == "e_above_hull":
        print("Using e_above_hull condition for unconditional sampling")
    for _ in range(args.num_samples):
        prompt = "Below is a description of a bulk material. "
        if args.conditions == "e_above_hull":
            prompt += "The energy above the convex hull is 0. "
        # if args.wyckoff:
        #     if args.prompt2:
        #         prompt += (
        #             "Generate a crystal structure with the chemical formula, space group,"
        #             "the lengths and angles of the lattice vectors, the number of sites, "
        #             "and the element type and coordinates and wyckoff positions for each atom within the lattice:\n"
        #         )
        #     else:
        #         prompt += (
        #             "Generate a crystal structure with the chemical formula, space group,"
        #             "the lengths and angles of the lattice vectors, the number of sites, "
        #             "and the element type and coordinates and wyckoff positions for each atom within the lattice. "
        #             "Enclose the crystal structure within <crystal> </crystal> tags.\n"
        #         )
        # else:
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)

    outputs = []
    progress_bar = tqdm(total=args.num_samples, desc="Generating samples")
    total_samples = 0
    while len(outputs) < args.num_samples:
        batch_prompts = prompts[len(outputs) : len(outputs) + args.batch_size]

        batch = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        total_samples += len(gen_strs)
        if total_samples < 1000:
            print(f"Total samples: {total_samples}")
        # Process the batch in parallel
        batch_outputs = process_batch_parallel(
            gen_strs,
            batch_prompts,
            args,
            max_workers=None,  # Uses CPU count by default
        )

        outputs.extend(batch_outputs)

        # Update the progress bar based on the length of outputs
        progress_bar.n = len(outputs)
        progress_bar.last_print_n = len(outputs)
        progress_bar.update(0)  # This forces the progress bar to update

    print("Total samples generated: ", total_samples)
    progress_bar.close()
    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)


condition_templates = {
    "pretty_formula": "The chemical formula is {pretty_formula}. ",
    "e_above_hull": "The energy above the convex hull is {e_above_hull}. ",
    "spacegroup_number": "The spacegroup number is {spacegroup_number}. ",
}


def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    conditions_data = pd.read_csv(args.conditions_file)
    required_columns = ["e_above_hull", "pretty_formula", "spacegroup_number"]
    available_columns = [
        col for col in required_columns if col in conditions_data.columns
    ]
    conditions_data = conditions_data[available_columns]
    print(conditions_data.head())

    if args.num_samples is None:
        conditions_data = conditions_data.to_dict(orient="records")
    else:
        conditions_data = conditions_data.sample(
            args.num_samples, replace=False
        ).to_dict(orient="records")

    conditions = args.conditions.split(",")

    prompts = []
    for d in conditions_data:
        prompt = "Below is a description of a bulk material. "
        for c in conditions:
            prompt += condition_templates[c].format(**d)

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)

    if args.num_samples is None:
        args.num_samples = len(prompts)

    outputs = []
    remaining_indices = deque(range(len(prompts)))  # Use deque for efficient tracking
    progress_bar = tqdm(total=args.num_samples, desc="Generating samples")
    total_samples = 0
    while len(outputs) < args.num_samples:
        batch_size = min(args.batch_size, len(remaining_indices))
        batch_indices = [remaining_indices.popleft() for _ in range(batch_size)]
        batch_prompts = [prompts[i] for i in batch_indices]

        batch = tokenizer(
            list(batch_prompts),
            padding=True,
            truncation=True,
            max_length=500,
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # print num of cpu cores avail for ProcessPoolExecutor
        print(f"Num of CPU cores avail: {os.cpu_count()}")
        # Process in parallel
        batch_outputs, failed_batch_indices = conditional_process_batch_parallel(
            gen_strs, batch_prompts, batch_indices, args
        )

        # Add successful results
        outputs.extend(batch_outputs)

        # Re-add failed indices for retry
        remaining_indices.extend(failed_batch_indices)

        # Update progress bar
        progress_bar.n = len(outputs)
        progress_bar.last_print_n = len(outputs)
        progress_bar.update(0)
        total_samples += len(batch_outputs)
        if total_samples < 10000:
            print(f"Total samples: {total_samples}")

    progress_bar.close()

    df = pd.DataFrame(outputs)
    # Save results
    if "spacegroup_number" in conditions:
        try:
            df = parse_sg_col(df)
        except Exception as e:
            print(f"Error parsing spacegroup from csv: {e}")

    df.to_csv(args.out_path, index=False)
    print(f"Total samples generated: {total_samples}")

    if "spacegroup_number" in conditions:
        try:
            args.out_path = args.out_path.replace(".csv", "")
            seperate_dfs_by_condition(args.out_path, "spacegroup_number")
        except Exception as e:
            print(f"Error parsing spacegroup from csv: {e}")


def infill_sample(args, start_crystal_cif=None):
    model, tokenizer = prepare_model_and_tokenizer(args)

    if start_crystal_cif is None:
        df = pd.read_csv(args.infill_file)
        idx = np.random.randint(len(df))
        start_crystal_cif = df["cif_str"][idx]

    print("Start crystal cif:")
    print(start_crystal_cif)

    prompts = []
    species_to_remove_list = []
    masked_crystal_strs = []
    for _ in range(args.num_samples):
        prompt = (
            "Below is a partial description of a bulk material where one "
            'element has been replaced with the string "[MASK]":\n'
        )

        structure = Structure.from_str(start_crystal_cif, fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)
        species_to_remove_list.append(species_to_remove)

        crystal_string = get_crystal_string(start_crystal_cif)

        partial_crystal_str = crystal_string.replace(species_to_remove, "[MASK]")
        masked_crystal_strs.append(partial_crystal_str)

        prompt = prompt + partial_crystal_str + "\n"

        prompt += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        prompts.append(prompt)

    assert args.batch_size == 1, "Batch size must be 1 for infill sampling"

    swap_table = make_swap_table(args.infill_constraint_tolerance)

    outputs = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        species_to_remove_batch = species_to_remove_list[i : i + args.batch_size]
        masked_crystals = masked_crystal_strs[i : i + args.batch_size]

        batch = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        possible_elems = [str(s) for s in swap_table[species_to_remove_batch[0]]]

        kwargs = {
            "do_sample": True,
            "max_new_tokens": 10,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        if args.infill_do_constraint:
            kwargs["bad_words_ids"] = [tokenizer.encode(s) for s in possible_elems]

        generate_ids = model.generate(
            **batch,
            **kwargs,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, species_to_remove, masked_crystal in zip(
            gen_strs, batch_prompts, species_to_remove_batch, masked_crystals
        ):
            new_element = gen_str.replace(prompt, "").split("\n")[0]

            print(f"Swap {species_to_remove} with {new_element}")

            orig_crys_str = masked_crystal.replace("[MASK]", species_to_remove)
            new_crys_str = masked_crystal.replace("[MASK]", new_element)

            try:
                new_cif = parse_fn(new_crys_str)
                _ = Structure.from_str(
                    new_cif, fmt="cif"
                )  # double check valid cif string
                original_cif = parse_fn(orig_crys_str)
            except Exception as e:
                print(e)
                continue

            sample = {
                "original_element": species_to_remove,
                "new_element": new_element,
                "original_crystal": original_cif,
                "new_crystal": new_cif,
                "model_name": args.model_name,
            }
            outputs.append(sample)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="llm_samples.csv")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument(
        "--conditions_file", type=str, default=""
    )  # "data/with_tags/test.csv"
    parser.add_argument(
        "--infill_file", type=str, default=""
    )  # "data/with_tags/test.csv"
    parser.add_argument("--infill_do_constraint", type=int, default=0)
    parser.add_argument("--infill_constraint_tolerance", type=float, default=0.1)
    parser.add_argument("--llama2", action="store_true", default=False)
    parser.add_argument("--qwen", action="store_true", default=False)
    parser.add_argument("--wyckoff", action="store_true", default=False)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--prompt2", action="store_true", default=False)
    parser.add_argument("--max_dir", action="store_true", default=False)
    parser.add_argument("--dpo", action="store_true", default=False)
    parser.add_argument("--use_fa2", action="store_true", default=False)

    args = parser.parse_args()

    if ".csv" in args.out_path:
        out_path = args.out_path
    else:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(args.out_path, f"samples_{i}.csv")
        args.out_path = out_path

    print("out_path: ", out_path)
    # Start timing
    start_time = time.time()

    if args.conditions_file:
        conditional_sample(args)
    elif args.infill_file:
        infill_sample(args)
    else:
        unconditional_sample(args)

    # Calculate and print execution time
    end_time = time.time()
    execution_minutes = (end_time - start_time) / 60
    print(f"Script execution time: {execution_minutes:.2f} minutes")
    print("Script execution time seconds: ", end_time - start_time)
