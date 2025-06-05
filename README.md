This repository contains the official implementation for our paper:

[**_PLaID++: A Preference-Aligned Language Model for Targeted Inorganic Materials Design_**](https://openreview.net/forum?id=JCkX2EgIrt)
(NeurIPS 2025, submitted)

> **Summary:**  
PLaID++ introduces a large language model (LLM), fine-tuned for stable and property-targeted inorganic crystal generation. Built upon Qwen-2.5 7B with a novel Wyckoff-based text encoding, PLaID++ generates crystal structures that are thermodynamically stable, unique, and novel at rates substantially higher than previous methods. We further align the model using Direct Preference Optimization (DPO), a reinforcement learning method guided by machine-learned interatomic potentials. PLaID++ achieves a ~20% higher S.U.N. (Stable, Unique, Novel) rate than prior work and robust conditional generation by space group, validated with DFT calculations. This demonstrates the promise of LLMs for targeted, efficient discovery of novel materials.

## Setup

First, create a conda environment and install the dependencies:

```
conda create -n plaid python=3.11
conda activate plaid
pip install -r requirements.txt
```

## Usage

Below are the main entry points for running the core workflows. For detailed options, please consult the script comments.

1. Supervised Fine-Tuning
To fine-tune Qwen-2.5 7B on crystal text representations (coordinate or Wyckoff):

```
python llm_finetune.py --run-name 7b-wyckoff-run-qwen --model 7b --batch-size 16 --fp4 --lr 5e-4 --qwen --teapot

```

2. Direct Preference Optimization (DPO) Fine-Tuning
To run DPO for preference-aligned RL across 3 iterations(see `scripts/qwenSGCombinedt2.sh` for the full script):

```
bash scripts/qwenSGCombinedt2.sh | tee logs/qwenSGCombinedt2.log
```

3. MLIP Evals: To evaluate our crystals, run
```
python evals/novelty.py
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

The full PLaID++ model is available under the `PLaID++/` directory.

## License

Most of PLaID++ is distributed under the CC BY 4.0 license. However, some components of the project are governed by different licenses: pymatgen is licensed under MIT, Hugging Face Transformers under Apache 2.0, and ASE under the GNU Lesser General Public License.
