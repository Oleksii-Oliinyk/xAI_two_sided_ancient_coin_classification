import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import SIDES


# catalogue.json key per side
DESCRIPTION_KEYS = {"Observe": "observe_description", "Reverse": "reverse_description"}

SEPARATORS = re.compile(r"[·,.]")
LIGATURE_MARK = "͡"  # combining double inverted breve used in legends, e.g. PIET͡AS


def _clean(text):
    text = text.replace(LIGATURE_MARK, "").lower()
    return re.sub(r"\s+", " ", text).strip()


class CatalogueFeatureExtractor():

    def __init__(self, catalogue_path, similarity_threshold=0.75):
        """
        catalogue_path:       catalogue.json created by data_split
        similarity_threshold: features with cosine similarity >= this count as "the same feature"
                              when counting in how many other classes a feature appears
        """
        with open(catalogue_path, encoding="utf-8") as f:
            self.catalogue = json.load(f)
        self.similarity_threshold = similarity_threshold

    def extract(self, text):
        """one description -> unique lower case features, split ONLY on '·', ',' and '.'"""
        features = []
        for piece in SEPARATORS.split(text):
            piece = _clean(piece)
            if piece and piece not in features:
                features.append(piece)
        return features

    def extract_all(self):
        """{side: {class_number: [feature, ...]}}"""
        return {
            side: {int(c): self.extract(info[DESCRIPTION_KEYS[side]]) for c, info in self.catalogue.items()}
            for side in SIDES
        }

    def compute_uniqueness(self, features):
        """
        For every feature compare it with the features of all OTHER classes on the same side:
          uniqueness      = 1 - max cosine similarity (char 3-5 grams TF-IDF), 1 = nothing similar anywhere
          similar_classes = other classes that have a feature with similarity >= similarity_threshold
        """
        rows = []
        for side, class_features in features.items():
            flat = [(c, f) for c, items in class_features.items() for f in items]
            texts = [f for _, f in flat]
            owners = np.array([c for c, _ in flat])

            vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
            similarity = cosine_similarity(vectors)

            for i, (class_number, feature) in enumerate(flat):
                other = owners != class_number
                other_similarity = np.where(other, similarity[i], -1.0)
                closest = int(other_similarity.argmax())

                similar = sorted({int(c) for c in owners[other & (similarity[i] >= self.similarity_threshold)]})
                rows.append({
                    "class_number": class_number,
                    "side": side,
                    "feature": feature,
                    "uniqueness": round(max(0.0, 1 - float(other_similarity[closest])), 4) + 0.0,
                    "similar_classes_count": len(similar),
                    "similar_classes": similar,
                    "closest_feature": texts[closest],
                    "closest_class": int(owners[closest]),
                })

        # per class and side: most unique first
        rows.sort(key=lambda r: (r["class_number"], SIDES.index(r["side"]), -r["uniqueness"]))
        rank = {}
        for row in rows:
            key = (row["class_number"], row["side"])
            rank[key] = rank.get(key, 0) + 1
            row["rank"] = rank[key]
        return rows

    def save_csv(self, rows, save_path):
        columns = ["class_number", "side", "rank", "feature", "uniqueness",
                   "similar_classes_count", "similar_classes", "closest_feature", "closest_class"]
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig so Excel shows special legend characters correctly
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "similar_classes": " ".join(map(str, row["similar_classes"]))})
        print(f"Saved {len(rows)} features to {save_path}")

    def save_json(self, rows, save_path):
        """
        {"<class number>": {"observe": [{"feature", "uniqueness", ...}, ...], "reverse": [...]}}
        features of every side are already sorted from most to least unique
        """
        result = {}
        for row in rows:
            side_features = result.setdefault(str(row["class_number"]), {s.lower(): [] for s in SIDES})
            side_features[row["side"].lower()].append({
                "feature": row["feature"],
                "uniqueness": row["uniqueness"],
                "similar_classes": row["similar_classes"],
                "closest_feature": row["closest_feature"],
                "closest_class": row["closest_class"],
            })

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved features of {len(result)} classes to {save_path}")
