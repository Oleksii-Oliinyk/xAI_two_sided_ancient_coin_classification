import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from src.constants import EXPLAINERS, MODEL_CONFIGS
from src.dataset.custom_dataset import get_base_transform
from src.xai.core.visualization import save_explanations_figure


@dataclass
class SidePrediction:
    side: str
    run: str
    class_number: int
    confidence: float
    top_k: list = field(default_factory=list)  # [(class_number, probability), ...]
    xai_path: str | None = None  # explanations figure for the final predicted class, if explainers were given


@dataclass
class CoinPrediction:
    class_number: int
    confidence: float
    observe: SidePrediction | None
    reverse: SidePrediction | None
    sides_agree: bool | None  # None when only one side was given
    catalogue: dict

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        lines = [
            f"Predicted class: {self.class_number} (confidence {self.confidence:.2%})",
        ]
        for side_prediction in (self.observe, self.reverse):
            if side_prediction is None:
                continue
            top_k = ", ".join(f"{c}: {p:.2%}" for c, p in side_prediction.top_k)
            lines.append(
                f"  {side_prediction.side:<8} -> class {side_prediction.class_number} "
                f"({side_prediction.confidence:.2%}) | top-{len(side_prediction.top_k)}: {top_k}"
            )
            if side_prediction.xai_path:
                lines.append(f"  {'':<8}    XAI: {side_prediction.xai_path}")
        if self.sides_agree is False:
            lines.append("  Warning: Observe and Reverse models disagree, combined prediction is used")

        info = self.catalogue
        lines += [
            "",
            f"Crawford number:     {info.get('crawford_number', '')}",
            f"Denomination:        {info.get('denomination', '')}",
            f"Date range:          {info.get('date_range', '')}",
            f"Issuer:              {info.get('issuer', '')}",
            f"Observe description: {info.get('observe_description', '')}",
            f"Reverse description: {info.get('reverse_description', '')}",
            f"Source:              {info.get('description_source', '')}",
        ]
        return "\n".join(lines)


class CoinInference():

    def __init__(
        self,
        observe_run=None,
        reverse_run=None,
        catalogue_path="dataset/data/catalogue.json",
        top_k=3,
        explainers=None,
        data_root="dataset/data",
        xai_dir="runs/xai",
    ):
        """
        observe_run:    run folder of a model trained with --side Observe (e.g. runs/resnet18_Observe_2026...)
        reverse_run:    run folder of a model trained with --side Reverse
        catalogue_path: catalogue.json created by data_split
        top_k:          how many best classes to report per side
        explainers:     names from EXPLAINERS (e.g. ["gradcam", "shap"]), every prediction then saves one
                        figure per side with all selected explanations. None / [] = no explanations
        data_root:      dataset splits, SHAP background images are taken from train/<side>
        xai_dir:        where explanation figures are saved
        """
        if observe_run is None and reverse_run is None:
            raise ValueError("At least one of observe_run / reverse_run is required")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_base_transform()
        self.top_k = top_k

        with open(catalogue_path, encoding="utf-8") as f:
            self.catalogue = json.load(f)
        # same numeric order as CoinDataset.classes -> model output i == class_numbers[i]
        self.class_numbers = sorted(int(c) for c in self.catalogue)

        self.runs = {"Observe": observe_run, "Reverse": reverse_run}
        self.models, self.model_names = {}, {}
        for side, run in self.runs.items():
            if run is not None:
                self.model_names[side], self.models[side] = self._load_model(run, side)

        unknown = [name for name in explainers or [] if name not in EXPLAINERS]
        if unknown:
            raise ValueError(f"Unknown explainers {unknown}, available: {list(EXPLAINERS)}")

        self.xai_dir = Path(xai_dir)
        # {side: {explainer name: explainer}}
        self.explainers = {
            side: {
                name: EXPLAINERS[name](
                    model=model,
                    device=self.device,
                    target_layer=MODEL_CONFIGS[self.model_names[side]]["target_layer"],
                    data_root=data_root,
                    side=side,
                )
                for name in explainers or []
            }
            for side, model in self.models.items()
        }

    def _load_model(self, run_dir, side):
        run_dir = Path(run_dir)
        # run folder is "<model_name>_<side>_<date>_<time>", model_name itself may contain "_"
        model_name, run_side, _, _ = run_dir.name.rsplit("_", 3)
        if run_side != side:
            raise ValueError(f"{run_dir} was trained on {run_side}, but is used for {side}")
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{model_name}' in run folder {run_dir}")

        model = MODEL_CONFIGS[model_name]["model"]()
        state_dict = torch.load(run_dir / "best_model.pth", map_location=self.device)
        model.load_state_dict(state_dict)
        return model_name, model.to(self.device).eval()

    def _load_image(self, image_path):
        """image_path: file path or an already opened PIL image (e.g. an upload from the web app)"""
        image = image_path if isinstance(image_path, Image.Image) else Image.open(image_path)
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def _predict_probs(self, side, image_tensor):
        return torch.softmax(self.models[side](image_tensor.unsqueeze(0).to(self.device)), dim=1)[0].cpu()

    def _explain(self, side, image_tensor, image_path, class_idx, probs):
        class_number = self.class_numbers[class_idx]
        explanations = {
            explainer.name: (explainer.explain(image_tensor, class_idx), explainer.signed)
            for explainer in self.explainers[side].values()
        }
        title = (f"{side} | {Path(self.runs[side]).name}\n"
                 f"Explanations for class {class_number} (p = {float(probs[class_idx]):.2%} by this side's model)")
        save_path = self.xai_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{side}_{Path(image_path).stem}_class{class_number}.png"
        return str(save_explanations_figure(image_tensor, explanations, title, save_path))

    def _side_prediction(self, side, probs):
        top_probs, top_idx = probs.topk(self.top_k)
        top_k = [(self.class_numbers[i], p) for i, p in zip(top_idx.tolist(), top_probs.tolist())]
        return SidePrediction(
            side=side,
            run=Path(self.runs[side]).name,
            class_number=top_k[0][0],
            confidence=top_k[0][1],
            top_k=top_k,
        )

    def predict(self, observe_image=None, reverse_image=None):
        images = {"Observe": observe_image, "Reverse": reverse_image}
        images = {side: path for side, path in images.items() if path is not None}
        if not images:
            raise ValueError("At least one of observe_image / reverse_image is required")

        tensors, probs, side_predictions = {}, {}, {}
        for side, image_path in images.items():
            if side not in self.models:
                raise ValueError(f"Got a {side} image, but no {side} model was loaded")
            tensors[side] = self._load_image(image_path)
            probs[side] = self._predict_probs(side, tensors[side])
            side_predictions[side] = self._side_prediction(side, probs[side])

        # both sides show the same coin -> average the class probabilities of both models
        combined = torch.stack(list(probs.values())).mean(dim=0)
        best_idx = int(combined.argmax())
        class_number = self.class_numbers[best_idx]

        # explain the final class (not each side's own top class) so both figures answer the same question
        for side in side_predictions:
            if self.explainers[side]:
                side_predictions[side].xai_path = self._explain(
                    side, tensors[side], images[side], best_idx, probs[side]
                )

        sides_agree = None
        if len(side_predictions) == 2:
            sides_agree = side_predictions["Observe"].class_number == side_predictions["Reverse"].class_number

        return CoinPrediction(
            class_number=class_number,
            confidence=float(combined[best_idx]),
            observe=side_predictions.get("Observe"),
            reverse=side_predictions.get("Reverse"),
            sides_agree=sides_agree,
            catalogue=self.catalogue[str(class_number)],
        )
