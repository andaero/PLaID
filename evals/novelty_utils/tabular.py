"""
This file depends on and heavily modifies code from Meta's flowllm repository, which is MIT-licensed.
The original license is preserved.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, get_args

import joblib
import pandas as pd
from pymatgen.core import Composition, Structure

from evals.novelty_utils.pandas_ import maybe_get_missing_columns
from evals.novelty_utils.pymatgen_ import COLUMNS_COMPUTATIONS

ValidStages = Literal["train", "val", "test"]
VALID_STAGES: Tuple[ValidStages, ...] = get_args(ValidStages)

ValidTabularDatasets = Literal["diffcsp_mp20", "cdvae_mp20", "alex_mp20"]
VALID_TABULAR_DATASETS: Tuple[ValidTabularDatasets, ...] = get_args(
    ValidTabularDatasets
)

trap = io.StringIO()


def get_tabular_dataset(name: ValidTabularDatasets) -> TabularDataset:
    if name == "diffcsp_mp20":
        return DiffCSP_MP20()
    elif name == "cdvae_mp20":
        return CDVAE_MP20()
    elif name == "alex_mp20":
        return Alex_MP20()
    else:
        raise ValueError()


@dataclass
class TabularDataset(ABC):
    root: Path
    train_unprocessed_path: Path
    val_unprocessed_path: Path
    test_unprocessed_path: Path
    train_processed_json: Path
    val_processed_json: Path
    test_processed_json: Path
    method: str
    valid_stages: Tuple[ValidStages, ...] = VALID_STAGES

    @abstractmethod
    def process(
        self,
        stage: ValidStages,
    ) -> None:
        raise NotImplementedError()

    def process_all(self) -> None:
        for stage in self.valid_stages:
            self.process(stage)

    def get_df(self, stage: ValidStages) -> pd.DataFrame:
        json = self.root / getattr(self, f"{stage}_processed_json")
        if not json.exists():
            self.process(stage)
        df = pd.read_json(json)
        df["source"] = stage
        df["method"] = self.method
        return df

    @property
    def train_df(self) -> pd.DataFrame:
        return self.get_df("train")

    @property
    def val_df(self) -> pd.DataFrame:
        return self.get_df("val")

    @property
    def test_df(self) -> pd.DataFrame:
        return self.get_df("test")


@dataclass
class DiffCSP_MP20(TabularDataset):
    root: Path = (Path(__file__).parents[2] / "data").resolve()
    train_unprocessed_path: Path = Path("mp_20/train.csv")
    val_unprocessed_path: Path = Path("mp_20/val.csv")
    test_unprocessed_path: Path = Path("mp_20/test.csv")
    train_processed_json: Path = Path("mp_20/train.json")
    val_processed_json: Path = Path("mp_20/val.json")
    test_processed_json: Path = Path("mp_20/test.json")
    method: str = "diffcsp_mp20"

    def process(
        self,
        stage: ValidStages,
    ) -> None:
        unprocessed_path = self.root / getattr(self, f"{stage}_unprocessed_path")
        processed_json = self.root / getattr(self, f"{stage}_processed_json")

        df = pd.read_csv(unprocessed_path)
        df["mp_id"] = df["material_id"]

        our_columns_computations = deepcopy(COLUMNS_COMPUTATIONS)
        our_columns_computations["composition"] = lambda ddf: ddf["pretty_formula"].map(
            lambda x: Composition(x).as_dict()
        )
        # create a copy of cif column called structure
        df["structure"] = df["cif"]
        df = maybe_get_missing_columns(df, our_columns_computations)

        # save
        df.to_json(processed_json, default_handler=str)
        print("done!")


@dataclass
class CDVAE_MP20(TabularDataset):
    root: Path = (Path(__file__).parents[2] / "remote/cdvae/data").resolve()
    train_unprocessed_path: Path = Path("mp_20/train.joblib")
    val_unprocessed_path: Path = Path("mp_20/val.joblib")
    test_unprocessed_path: Path = Path("mp_20/test.joblib")
    train_processed_json: Path = Path("mp_20/train.json")
    val_processed_json: Path = Path("mp_20/val.json")
    test_processed_json: Path = Path("mp_20/test.json")
    method: str = "cdvae_mp20"

    def process(
        self,
        stage: ValidStages,
    ) -> None:
        unprocessed_path = self.root / getattr(self, f"{stage}_unprocessed_path")
        processed_json = self.root / getattr(self, f"{stage}_processed_json")

        print("computing df...")
        cached_data = joblib.load(unprocessed_path)
        rows = []
        for row in cached_data:
            with redirect_stdout(trap):
                structure = Structure.from_str(row["cif"], fmt="cif")
            rows.append(
                {
                    "mp_id": row["mp_id"],
                    "num_sites": structure.num_sites,
                }
            )
        df = pd.DataFrame.from_records(rows)
        df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)

        # save
        df.to_json(processed_json)
        print("done!")


@dataclass
class Alex_MP20(TabularDataset):
    root: Path = (Path(__file__).parents[2] / "data").resolve()
    train_unprocessed_path: Path = Path("alex_mp_20/train.csv")
    val_unprocessed_path: Path = Path("alex_mp_20/val.csv")
    test_unprocessed_path: Path = None
    train_processed_json: Path = Path("alex_mp_20/train.json")
    val_processed_json: Path = Path("alex_mp_20/val.json")
    test_processed_json: Path = None
    method: str = "alex_mp20"
    valid_stages: Tuple[ValidStages, ...] = ("train", "val")

    def process(
        self,
        stage: ValidStages,
    ) -> None:
        if stage == "test":
            print("no test data for alex_mp20")
            return
        unprocessed_path = self.root / getattr(self, f"{stage}_unprocessed_path")
        processed_json = self.root / getattr(self, f"{stage}_processed_json")

        df = pd.read_csv(unprocessed_path)
        df["mp_id"] = df["material_id"]
        print(df.columns)
        # rename reduced_formula to pretty_formula
        df = df.rename(columns={"reduced_formula": "pretty_formula"})
        # drop rows with no composition
        # print the number of rows before and after
        print(f"Number of rows before: {len(df)}")
        df = df[df["pretty_formula"].notna()]
        print(f"Number of rows after: {len(df)}")
        our_columns_computations = deepcopy(COLUMNS_COMPUTATIONS)
        our_columns_computations["composition"] = lambda ddf: ddf["pretty_formula"].map(
            lambda x: Composition(x).as_dict()
        )
        df = maybe_get_missing_columns(df, our_columns_computations)

        # save
        df.to_json(processed_json, default_handler=str)
        print("done!")


if __name__ == "__main__":
    tds = DiffCSP_MP20()
    tds.process_all()
