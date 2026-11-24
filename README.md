# scPertEcho

scPertEcho predicts how genes respond to a human genetic perturbation in a specified cell state. It encodes the perturbed gene, expression in untreated cells and each output gene. A previous measurement of the same perturbation can refine the forecast. The toolkit also assigns two anonymous response profiles to two candidate perturbations.

## Install

Use Python 3.10 or later. For prediction and matching, install the package in a virtual environment.

```sh
git clone https://github.com/Qiu-Shizheng/scPertEcho.git
cd scPertEcho
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[inference]'
scpertecho --help
```

For profile matching alone, `python -m pip install .` installs NumPy and the command-line tools. Prediction runs on CPU with PyTorch. A versioned wheel, software archive, fitted weights and input transforms are available in [release 1.0.0](https://github.com/Qiu-Shizheng/scPertEcho/releases/tag/v1.0.0).

## 1. Identify two anonymous profiles

Run this example from the cloned repository. It uses invented expression changes and runs without model downloads.

```sh
scpertecho match \
  --forecasts examples/matching/forecasts.csv \
  --observed examples/matching/observations.csv \
  --panel examples/matching/panel.csv \
  --exclude examples/matching/screen_targets.csv
```

The assignment is `candidate-A → sample-2` and `candidate-B → sample-1`. The total squared errors for the direct and swapped assignments are 16 and 0, respectively. Three downstream genes contribute after excluding one directly targeted gene.

For your data, supply two rows in each table, with distinct `sample_id` values and columns named by gene. Predictions and measurements must describe signed changes from untreated controls in the same units. Predefine the genes that can be measured in both samples and list them under `canonical_gene` in the panel CSV. List every gene targeted in the screening library in the exclusion CSV. A header-only exclusion file is valid when the common panel already omits all targeted genes. Column order can differ between files. Equal assignment costs return `"tie": true` and `"assignment": null`.

## 2. Run the frozen network and measured-record correction

Download the project's fitted tensors (25 arrays with 1,298,528 learned parameters) and fitted input transforms once.

```sh
base=https://github.com/Qiu-Shizheng/scPertEcho/releases/download/v1.0.0
curl -fLO "$base/scPertEcho-weights-1.0.0.npz"
curl -fLO "$base/scPertEcho-encoding-1.0.0.npz"
curl -fLO "$base/scPertEcho-SHA256SUMS.txt"
sha256sum --ignore-missing -c scPertEcho-SHA256SUMS.txt

scpertecho demo --weights scPertEcho-weights-1.0.0.npz --output demo.npz
```

This demo uses invented numerical features, two target rows, four output genes, a reference expression scale and two measured-record examples. It runs both learned output heads and the published record correction. The output contains `rna_residual` (standardised response) and `de_logit_residual` (change in differential-expression log odds), followed by calibrated `response` and `de_score`. With release 1.0.0, the first returned response is **0.027071** and the first differential-expression score is **0.257379** (rounded to six places). These numbers describe the invented example.

## 3. Forecast from untreated expression and target names

Download the two files of the Arc Institute PIE gene-text resource directly from the upstream repository. The frozen preprocessing expects the revision below. The published fitted transform maps its 3,072-dimensional gene descriptions into the model's 128-dimensional input coordinates.

```sh
mkdir -p gene_text
src=https://huggingface.co/datasets/arcinstitute/PIE_sources/resolve/fb624a5158a7f37dfa13a015cdacadb985549d26/gene_text
curl -fL "$src/meta.json" -o gene_text/meta.json
curl -fL "$src/embeddings.npy" -o gene_text/embeddings.npy

python examples/make_control_demo.py scPertEcho-encoding-1.0.0.npz control-demo
scpertecho encode --encoding scPertEcho-encoding-1.0.0.npz \
  --annotations gene_text --controls control-demo/controls.csv \
  --targets control-demo/targets.csv --genes control-demo/genes.csv \
  --output encoded.npz
scpertecho predict --weights scPertEcho-weights-1.0.0.npz \
  --inputs encoded.npz --output forecast.npz
python -c "import numpy as np; p=np.load('forecast.npz'); print(p['row_id'], p['gene_id']); print(p['rna_residual'])"
```

The example prepares an invented untreated profile, two developmental target symbols (GATA6 and SOX17) and five named downstream genes. It returns a 2 × 5 matrix for each learned head. The first standardised RNA residual is about **0.0035145**. To analyse your experiment, replace the three CSVs.

| File | Required columns | Biological meaning |
| --- | --- | --- |
| `controls.csv` | `canonical_gene`, one column per context | For each gene, the mean across untreated cells of per-cell log(1 + counts per 10,000). At least 410 of the 512 trained anchor genes are required. |
| `targets.csv` | `row_id,target,context` | A unique name for each perturbation–context combination, a human gene symbol and the matching control column. |
| `genes.csv` | `canonical_gene` | Ordered, distinct Ensembl IDs of output genes on the trained gene axis. |

The public encoder supports the mean-cell log-normalised assay. The frozen transform fills each missing control anchor with zero on the normalised scale. Unmatched gene-description symbols receive the fallback zero annotation. `encoded.npz` records annotation coverage so users can inspect it. The basic prediction gives network residuals. To predict signed expression changes in a reference scale, add the calibration and optional records described below.

## 4. Supply calibrated expression scales and previous measurements

```sh
scpertecho predict --weights scPertEcho-weights-1.0.0.npz \
  --inputs encoded.npz --calibration calibration.npz \
  --records records.npz --output assisted.npz
```

Calibration uses, for each target row and output gene, a reference mean response, a positive standard deviation and a prior log odds of differential expression. Use values estimated from comparable reference screens. The full predictor adds the scaled neural residual to the reference mean and adds the differential-expression residual to the prior log odds before the sigmoid. Optional compatible measurements then update both outputs. For a user-supplied reference collection, the forecast reflects that collection. [Input schema and a worked construction](docs/input-format.md).

The [model card](docs/model-card.md) describes the fitted weights, training data, numerical verification and scope of the reported biological results.

## Python API

```python
from scpertecho.model import load_model, predict
from scpertecho.matching import match_profiles

model = load_model("scPertEcho-weights-1.0.0.npz")
scores = predict(model, "encoded.npz")
# Optional: predict(model, "encoded.npz", "calibration.npz", "records.npz")
print(scores["rna_residual"].shape)

assignment = match_profiles([[1, -1], [-1, 1]], [[-1, 1], [1, -1]],
                            ["A", "B"], ["sample-1", "sample-2"], ["g1", "g2"], [])
print(assignment["assignment"])  # {'A': 'sample-2', 'B': 'sample-1'}
```
