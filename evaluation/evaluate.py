import argparse
import json
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Evaluate a labeled Ultralytics dataset.")
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True, help="Dataset YAML")
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--confidence", type=float, default=0.001)
parser.add_argument("--project", default="evaluation/results")
parser.add_argument("--name", default="overall")
args = parser.parse_args()

model = YOLO(args.model)
metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                    conf=args.confidence, project=args.project, name=args.name, plots=True)
names = getattr(metrics, "names", model.names)
summary = {
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
    "mAP75": float(metrics.box.map75),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "per_class_mAP50_95": {str(name): float(value) for name, value in zip(names, metrics.box.maps)},
    "note": "Run separately on labeled distance/angle subsets with distinct --name values."
}
output = Path(args.project) / args.name / "metrics_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("Saved:", output)
