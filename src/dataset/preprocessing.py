import json
import random
import shutil
from pathlib import Path

import pandas as pd

from src.constants import CLASS_NUMBER, IMAGE_EXTENSIONS, SIDES

CATALOGUE_FILE = "catalogue.json"

# excel column -> key in catalogue.json
CATALOGUE_COLUMNS = {
    "Crawford number": "crawford_number",
    "Denomination": "denomination",
    "Date range": "date_range",
    "Issuer": "issuer",
    "Observe description": "observe_description",
    "Reverse description": "reverse_description",
    "Description source": "description_source",
}


class DatasetPreprocessing():

    def __init__(self, data_root, output_folder, catalogue_path=None, split_ratio=0.15, seed=42):
        """
        data_root:      folder containing 'Observe' and 'Reverse', each with one sub-folder per class
        output_folder:  where train/val/test folders and catalogue.json will be created
        catalogue_path: excel catalogue with one row per class, defaults to data_root/RRC60_catalogue.xlsx
        split_ratio:    fraction of EACH class that goes to val and to test (train gets the rest)
        """
        if not 0 < split_ratio < 0.5:
            raise ValueError("split_ratio must be in (0, 0.5) so train keeps at least half of every class")

        self.data_root = Path(data_root)
        self.output_folder = Path(output_folder)
        self.catalogue_path = Path(catalogue_path) if catalogue_path else self.data_root / "RRC60_catalogue.xlsx"
        self.split_ratio = split_ratio
        self.seed = seed

    def _list_images(self, class_dir):
        return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    def _split_class(self, files, rng):
        files = files[:]
        rng.shuffle(files)

        n = len(files)
        n_val = round(n * self.split_ratio)
        n_test = round(n * self.split_ratio)
        # guarantee every split gets at least one sample when the class is big enough
        if n >= 3:
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)

        return {
            "val": files[:n_val],
            "test": files[n_val:n_val + n_test],
            "train": files[n_val + n_test:],
        }

    def data_split(self):
        """
        Divide every class into train/val/test without touching the disk.
        Returns {(side, class_name): {"train": [paths], "val": [paths], "test": [paths]}}
        """
        rng = random.Random(self.seed)
        splits = {}

        for side in SIDES:
            side_dir = self.data_root / side
            if not side_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {side_dir}")

            class_dirs = sorted(d for d in side_dir.iterdir() if d.is_dir())
            if len(class_dirs) != CLASS_NUMBER:
                print(f"Warning: {side} has {len(class_dirs)} classes, expected {CLASS_NUMBER}")

            for class_dir in class_dirs:
                splits[(side, class_dir.name)] = self._split_class(self._list_images(class_dir), rng)

        self._print_stats(splits)
        return splits

    def save_split(self, splits):
        """
        Copy files into output_folder/{train,val,test}/{Observe,Reverse}/<class>/
        """
        for (side, class_name), class_splits in splits.items():
            for split_name, split_files in class_splits.items():
                target_dir = self.output_folder / split_name / side / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                for f in split_files:
                    shutil.copy2(f, target_dir / f.name)

        print(f"Saved split to {self.output_folder}")

    def save_catalogue(self):
        """
        Convert the excel catalogue into output_folder/catalogue.json:
        {"<class number>": {"crawford_number": ..., "denomination": ..., ...}}
        """
        df = pd.read_excel(self.catalogue_path)

        missing = [c for c in ["Class number", *CATALOGUE_COLUMNS] if c not in df.columns]
        if missing:
            raise ValueError(f"Catalogue {self.catalogue_path} is missing columns: {missing}")

        catalogue = {}
        for row in df.to_dict("records"):
            catalogue[str(int(row["Class number"]))] = {
                key: str(row[column]).strip() if pd.notna(row[column]) else ""
                for column, key in CATALOGUE_COLUMNS.items()
            }

        if len(catalogue) != CLASS_NUMBER:
            print(f"Warning: catalogue has {len(catalogue)} classes, expected {CLASS_NUMBER}")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        save_path = self.output_folder / CATALOGUE_FILE
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, ensure_ascii=False, indent=2)

        print(f"Saved catalogue ({len(catalogue)} classes) to {save_path}")
        return catalogue

    def _print_stats(self, splits):
        print(f"{'side':<8} {'class':<30} {'train':>6} {'val':>6} {'test':>6}")
        totals = {"train": 0, "val": 0, "test": 0}
        for (side, cls), class_splits in splits.items():
            counts = {k: len(v) for k, v in class_splits.items()}
            print(f"{side:<8} {cls:<30} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6}")
            for k in totals:
                totals[k] += counts[k]
        print(f"{'TOTAL':<39} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6}")
