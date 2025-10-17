# PLaID++

This repository contains the official implementation for our paper: [**_PLaID++: A Preference-Aligned Language Model for Targeted Inorganic Materials Design_**](https://arxiv.org/pdf/2509.07150), by [Andy Xu](https://www.linkedin.com/in/andyxuai/), [Rohan Desai](https://www.rohandesai.xyz), [Larry Wang](https://www.linkedin.com/in/larwang314/), [Gabriel Hope](https://www.linkedin.com/in/gabriel-hope-87472542/), and [Ethan Ritz](https://www.linkedin.com/in/ethan-ritz-2bba69382/) (ICLR 2026, submitted).

## Summary

PLaID++ introduces an LLM fine-tuned for stable and property-targeted inorganic crystal generation. PLaID++ achieves a **~50% higher S.U.N.** (Stable, Unique, Novel) rate than prior work and robust conditional generation by space group though:
1. Leveraging a novel Wyckoff-based text encoding
2. Aligning the model using Direct Preference Optimization (DPO), an RL method guided by machine-learned interatomic potentials
3. Unified training across conditional and unconditional generation tasks

![plaid_architecture_diagram](https://arxiv.org/html/2509.07150v1/Figures/plaid++_diagram.png)

## Setup

First, create an environment and install the dependencies using uv:

```
uv venv plaid
source plaid/bin/activate
uv sync
```

## Usage

Below are the main entry points for running the core workflows. For detailed options, please consult the script comments.

1. Supervised Fine-Tuning
To fine-tune Qwen-2.5 7B on crystal text representations (coordinate or Wyckoff):

```
python llm_finetune.py --run-name 7b-wyckoff-run-qwen --model 7b --batch-size 16 --fp4 --lr 5e-4 --qwen
```

2. Direct Preference Optimization (DPO) Fine-Tuning
To run DPO for preference-aligned RL across 7 iterations(see `scripts/plaid_dpo.sh` for the full script).

```
bash scripts/plaid_dpo.sh | tee logs/plaid_dpo.log
```

3. MLIP Evals: To evaluate our crystals, run
```
python evals/novelty.py "${results.csv}" test.json "${model_name}" --sun_out "${sun_output.csv}"  # unconditional eval
bash run_novelty.sh qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.1 0.1 esen  # conditional eval
```

4. Visualizations
To plot space group, stability, and other metrics:

```
python visualizations/histogram_ehull.py
python visualizations/conditional_sg_bar_graph.py
```

5. DFT Pipeline
We run DFT on 1000 crystal structures sampled from the final PLaID++ flagship model. We've included scripts to help prepare the necessary configuration files to run DFT using vasp in directories corresponding to each crystal. 
```
python dft/LLM_dft_create_inputs.py
```

To compute corrected energy above hull from DFT outputs (namely through vasprun.xml files):
```
python dft/ehull_correction_newest.py
```

## Data
The `data/` directory contains the primary dataset splits.

For energy-above-hull (Ehull) evaluation or DFT validation, follow the instructions and links in the paper for any required external data.

Samples for unconditional and conditional generation used for the final model results in the main paper are available in the `results/` directory.

## Model

[The full PLaID++ model is available on HuggingFace](https://huggingface.co/HOPE-Lab-HMC/PLaID).

## Citation

[Arxiv Link](https://arxiv.org/pdf/2509.07150)
```
@article{xu2025plaid++,
  title={PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design},
  author={Xu, Andy and Desai, Rohan and Wang, Larry and Hope, Gabriel and Ritz, Ethan},
  journal={arXiv preprint arXiv:2509.07150},
  year={2025}
}
```

## License

Most of PLaID++ is distributed under the CC BY 4.0 license. However, some components of the project are governed by different licenses: pymatgen is licensed under MIT, Hugging Face Transformers under Apache 2.0, and ASE under the GNU Lesser General Public License.
