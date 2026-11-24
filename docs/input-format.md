# Numerical inputs

A row represents one perturbed target in one untreated context. The toolkit aligns rows and genes by their declared identifiers and checks their order. Store text arrays as NumPy Unicode and numerical arrays as finite float32 or integer values. The input format uses `np.savez` and loads with `allow_pickle=False`.

| Encoded array | Shape | Content |
| --- | --- | --- |
| `row_id`, `gene_id` | T, G | Unique row and gene labels in result order |
| `target_features` | T × 128 | Target-gene annotation in the frozen projection |
| `control_anchors` | T × 512 | Normalised untreated expression in the trained anchor order |
| `gene_features` | G × 128 | Output-gene annotation in the same projection |
| `gene_index` | G | Integer index into the learned gene embedding. Use the public encoder to map Ensembl IDs. |
| `view` | T | The public expression-control encoder supplies 0. Index 1 is reserved for inputs using the original primary-cell assay preprocessing. |

Fitted preprocessing coefficients are in `scPertEcho-encoding-1.0.0.npz`. The arrays `annotation_center`, `projection` and `annotation_scale` define the annotation encoder. For an original gene vector `e`, compute `clip(((e − center) @ projection.T) / scale, −8, 8)`. Standardise available untreated anchors using `anchor_center` and `anchor_scale`, then clip to [−6, 6]. The ordered `output_gene` and `symbol` arrays define the indices.

## Calibrated forecasts

Use reference observations on the same signed response scale as the intended destination. `response_mean` and `response_sd` are the estimated typical displacement and its scale in a comparable reference context. `prior_de_log_odds` gives the reference prior for differential expression. Each matrix has shape T × G. The original evaluation estimated these quantities across reference screens. Supplying different reference values changes the calibrated forecast.

In the example below, `mean`, `sd` and `log_odds` come from your reference estimates. For a single reference shared by all target rows, broadcast the gene vector to T × G and copy it.

```python
import numpy as np

with np.load("encoded.npz", allow_pickle=False) as query:
    rows, genes = query["row_id"], query["gene_id"]
# For example, use np.broadcast_to(mean_by_gene, (len(rows), len(genes))).copy().
np.savez("calibration.npz", row_id=rows, gene_id=genes,
         response_mean=mean, response_sd=sd, prior_de_log_odds=log_odds)
```

## Same-target records

Each of the R records comes from another measurement of the same target, selected without the anonymous destination responses. If using the study's selection rule, keep measurements with the comparable assay for the RNA head and experiments with differential-expression calls for the DE head. Within each modality, prefer same-study eligible records when available. A record may serve only one head. Supply the selected rows and their masks explicitly.

| Record array | Shape | Content |
| --- | --- | --- |
| `target_row` | R | Forecast row for each reference. A row ID may repeat. |
| `gene_id` | G | Same order as the forecast panel. |
| `distance` | R | Nonnegative mean squared difference between standardised untreated anchor profiles over available common anchors. |
| `rna_eligible`, `de_eligible` | R | Boolean flags selecting records for each output. |
| `rna_z` | R × G | Signed change minus the source-context mean, divided by the source-context standard deviation. Use NaN for a missing measurement. |
| `de_label` | R × G | Measured differential-expression call, 0 or 1. Use NaN for a missing call. |

For each target row and modality, the per-record weight is `exp(−(distance − minimum eligible distance) / 0.6)`. Available measurements contribute separately for each gene. The RNA correction uses the clipped source z score times the destination scale. The DE correction uses the observed call minus the model's calibrated DE score. In each head, multiply the sum of weighted corrections by 0.6 and divide by the available sum of weights plus two. A target with no eligible measurement retains its calibrated forecast.

The following fully specified example uses invented values to show both missing-gene masking and multiple records. Run it after the two-target, five-gene tutorial.

```python
import numpy as np
with np.load("encoded.npz") as query:
    rows, genes = query["row_id"], query["gene_id"]
shape = (len(rows), len(genes))
np.savez("calibration.npz", row_id=rows, gene_id=genes,
         response_mean=np.zeros(shape), response_sd=np.ones(shape),
         prior_de_log_odds=np.full(shape, -2.0))
z = np.zeros((2, len(genes)), dtype=np.float32)
z[:, 0] = [0.5, 0.8]
z[0, -1] = np.nan
label = np.full_like(z, np.nan)
label[0, 0] = 1
np.savez("records.npz", gene_id=genes, target_row=rows[[0, 0]],
         distance=np.array([0.2, 0.7]), rna_eligible=np.array([True, True]),
         de_eligible=np.array([True, False]), rna_z=z, de_label=label)
```
