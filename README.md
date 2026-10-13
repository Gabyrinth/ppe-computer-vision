# Windows Construction Site PPE Surveillance

## 1. Requirements
- Windows 10/11, Python 3.10 or 3.11 (64-bit recommended)
- Webcam
- The repository's `best.pt` model file

## 2. Setup in PowerShell
Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install 64-bit Python 3.10 or 3.11 from python.org and enable the Python Launcher during installation.

## 3. Add weights and run
Place the model here:
```text
models\best.pt
```
Run:
```powershell
python main.py
```
Press `q` in the preview window to exit. If camera 0 is unavailable:
```powershell
python main.py --camera 1
```

## 4. Adjust boundary and zone
Line height is a fraction of the image:
```powershell
python main.py --line-y 0.60
```
Zone coordinates are normalized x,y pairs from 0 to 1:
```powershell
python main.py --zone "0.10,0.20;0.45,0.20;0.45,0.85;0.10,0.85"
```

## 5. Outputs
- `logs\events.csv`: alert event records and snapshot paths.
- `logs\snapshots\`: JPEG snapshots for emitted alerts.
- `logs\compliance_over_time.csv`: compliance state changes per tracked person.
- Alerts: PPE violation, line crossing, and non-compliant restricted-zone entry.

## 6. Model labels
The program prints `model.names` at startup. Confirm they match labels in `PPE` in `main.py`. The system draws only the person box and associates PPE detections using box overlap. It uses explicit negative classes such as `NO-Hardhat` and `NO-Safety Vest`. Green means no explicit violation was detected, not guaranteed compliance. If the model does not contain explicit negative PPE classes, absence of a positive PPE detection is not enough to infer missing PPE.

## 7. Evaluation
Use a labeled YOLO dataset YAML:
```powershell
python evaluation\evaluate.py --model models\best.pt --data C:\path\to\data.yaml --split test --name overall
```
For distance/angle evaluation, prepare labeled subsets (near/mid/far and frontal/oblique) and run separately with distinct `--name` values. False-alarm rate requires negative-only footage and exposure normalization, such as false alarms/minute or per 1,000 frames. Do not use raw false-positive counts as a rate.

Prototype only. Validate on representative site footage and retain human oversight.
