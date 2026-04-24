"""
Modal wrapper for e_above_hull_batched.py.

Splits an input CSV across N Modal H200 workers, each running the full
relaxation + e_above_hull pipeline, then stitches results locally.

One-time volume setup:
    modal volume create plaid-data
    modal volume put plaid-data evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt /evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt
    modal volume put plaid-data evals/OCPCalcs/esen_30m_oam.pt /evals/OCPCalcs/esen_30m_oam.pt
    modal volume put plaid-data evals/2023-02-07-ppd-mp.pkl.gz /evals/2023-02-07-ppd-mp.pkl.gz

Usage:
    uv run modal run evals/e_above_hull_modal.py --filename data/foo.csv --num-workers 5
"""

import io
from pathlib import Path

import modal
import numpy as np
import pandas as pd

PROJECT_ROOT = "/root/project"
DATA_MOUNT = "/data"

app = modal.App("plaid-ehull")

volume = modal.Volume.from_name("plaid-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        "pip install torch-scatter==2.1.2 torch-sparse==0.6.18 "
        "-f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )
    .pip_install(
        "fairchem-core==1.10.0",
        "torch-sim-atomistic>=0.3.0",
        "pymatgen==2025.1.9",
        "pandas",
        "numpy==1.26.4",
        "ase",
        "smact==3.1.0",
        "tqdm",
        "scipy",
        "matminer",
        "p-tqdm",
    )
    .add_local_file("evals/__init__.py", remote_path=f"{PROJECT_ROOT}/evals/__init__.py")
    .add_local_file("evals/e_above_hull_batched.py", remote_path=f"{PROJECT_ROOT}/evals/e_above_hull_batched.py")
    .add_local_file("evals/llm_utils.py", remote_path=f"{PROJECT_ROOT}/evals/llm_utils.py")
    .add_local_file("evals/eval_util.py", remote_path=f"{PROJECT_ROOT}/evals/eval_util.py")
)


@app.function(
    image=image,
    gpu="H200",
    volumes={DATA_MOUNT: volume},
    timeout=4800,
)
def process_chunk(
    chunk_csv_bytes: bytes,
    chunk_index: int,
    relaxer_type: str,
    out_folder: str | None,
) -> bytes:
    import os
    import sys
    import tempfile
    import warnings

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    # Symlink volume data files into the project tree so existing relative paths work.
    # Volume layout: /data/evals/OCPCalcs/*.pt and /data/evals/2023-02-07-ppd-mp.pkl.gz
    os.makedirs(f"{PROJECT_ROOT}/evals/OCPCalcs", exist_ok=True)
    for src, dst in [
        (
            f"{DATA_MOUNT}/evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt",
            f"{PROJECT_ROOT}/evals/OCPCalcs/eqV2_86M_omat_mp_salex.pt",
        ),
        (
            f"{DATA_MOUNT}/evals/OCPCalcs/esen_30m_oam.pt",
            f"{PROJECT_ROOT}/evals/OCPCalcs/esen_30m_oam.pt",
        ),
        (
            f"{DATA_MOUNT}/evals/2023-02-07-ppd-mp.pkl.gz",
            f"{PROJECT_ROOT}/evals/2023-02-07-ppd-mp.pkl.gz",
        ),
    ]:
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    from evals.e_above_hull_batched import label_energies_batched, get_e_above_hull

    # Write chunk to a temp CSV
    with tempfile.NamedTemporaryFile(
        suffix=".csv", dir="/tmp", delete=False, mode="wb"
    ) as f:
        f.write(chunk_csv_bytes)
        chunk_path = f.name

    print(f"[Worker {chunk_index}] Processing chunk with {len(chunk_csv_bytes)} bytes")

    try:
        # Stage 1: GPU-batched relaxation
        relaxed_path = label_energies_batched(
            filename=Path(chunk_path),
            out_folder=out_folder,
            relaxer_type=relaxer_type,
            num_structures=None,
        )

        # Stage 2: e_above_hull computation
        warnings.filterwarnings("ignore")
        get_e_above_hull(relaxed_path)

        # Read back the final ehull results
        ehull_path = relaxed_path.replace("_relaxed_energy.csv", "_ehull_results.csv")
        with open(ehull_path, "rb") as f:
            result = f.read()

        print(f"[Worker {chunk_index}] Done, returning {len(result)} bytes")
        return result
    except Exception as e:
        print(f"[Worker {chunk_index}] FAILED: {e}")
        # Return header-only CSV so stitching skips this chunk gracefully
        header_end = chunk_csv_bytes.index(b"\n") + 1
        return chunk_csv_bytes[:header_end]


@app.local_entrypoint()
def main(
    filename: str,
    relaxer: str = "eqv2_batched",
    out_folder: str = None,
    num_structures: int = None,
    num_workers: int = 5,
    bandgap: bool = False,
):
    df = pd.read_csv(filename)
    total = len(df)

    if num_structures is not None:
        df = df.head(num_structures)

    if bandgap and out_folder is None:
        out_folder = "bandgap"

    # Split into chunks
    chunks = np.array_split(df, num_workers)
    chunk_bytes = [c.to_csv(index=False).encode("utf-8") for c in chunks]

    print(f"Splitting {len(df)} structures across {num_workers} workers")
    for i, c in enumerate(chunks):
        print(f"  Worker {i}: {len(c)} structures")

    # Fan out to Modal
    results = list(
        process_chunk.map(
            chunk_bytes,
            range(num_workers),
            [relaxer] * num_workers,
            [out_folder] * num_workers,
        )
    )

    # Stitch results
    dfs = []
    for i, result_bytes in enumerate(results):
        if result_bytes:
            chunk_df = pd.read_csv(io.BytesIO(result_bytes))
            if len(chunk_df) == 0:
                print(f"  Worker {i} returned no data rows (failed, skipped)")
            else:
                dfs.append(chunk_df)
        else:
            print(f"  Worker {i} returned empty result (skipped)")

    if not dfs:
        print("All workers returned empty results. No output written.")
        return

    final = pd.concat(dfs, ignore_index=True)

    # Write final output
    stem = Path(filename).stem
    relaxer_name = relaxer.replace("_batched", "")
    if out_folder:
        out_path = (
            Path("evals/results")
            / out_folder
            / f"{stem}_{relaxer_name}_ehull_results.csv"
        )
    else:
        out_path = Path("evals/results") / f"{stem}_{relaxer_name}_ehull_results.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"Final results ({len(final)} rows from {total} total): {out_path}")
