# Model card

scPertEcho estimates population-average responses to human genetic perturbations. It was trained on 23,011 target–background profiles from 20 cellular backgrounds using seed 42. Its 1,298,528 trainable parameters form a target-gene encoder, untreated-state encoder, state gate, target–state fusion, output-gene representation and separate RNA and differential-expression heads. The two numerical assets in release 1.0.0 contain learned weights, fitted transformation coefficients and gene identifiers. The files load as NumPy arrays without pickle.

## Interpretation

The network returns standardised RNA residuals and differential-expression logit residuals. Gene-specific reference moments place these outputs on the response and score scales. An optional additional measurement of the same target supplies an empirical correction. The resulting differential-expression score ranks genes. Its calibration depends on the reference assay and the labels used for fitting.

The main biological evaluation asked whether two predicted profiles distinguish the responses to two specified perturbations in independent experiments. The published model with its full reference collection attained 63.1% mean identification across five studies and 65.5% across three human developmental studies. The 50% chance level applies to this two-candidate task. With the same measured correction, the strongest comparator on the three-study panel reached 65.2%. Agreement with the ordering of individual transcript changes was weaker. A calibrated run with a different reference collection constitutes a different experimental setting.

The public toolkit provides frozen weights, the fitted gene/anchor transforms and the RNA and DE record-integration formula. Supply your own authorised records and calibration tables to use the measured-reference mode. Each input and output has a row and gene axis, so assay settings and comparisons can be recorded alongside results.

## Verification

An installable wheel was tested against frozen numerical outputs from an external K562 chronic myeloid leukaemia screen. For three perturbations across 513 genes, the test encoded actual untreated expression, loaded all 25 parameter tensors, applied eight same-target reference profiles and compared both calibrated output heads to the archived predictions. Maximum absolute differences were below 6 × 10⁻⁸ for the signed responses and 5 × 10⁻⁸ for the DE scores on this panel. This is an implementation agreement check. The public example uses invented records and controls.

The application-level examples and their expected numerical outputs are in [the validation record](validation.json). Small platform-dependent differences can occur in floating-point matrix products.

## Data and attribution

The fitted model used the Replogle–Nadig, Jiang, X-Atlas/Orion, Arc H1 and Zhu primary T-cell screens. The public encoder reads pinned gene-description embeddings from Arc's PIE sources. Their dataset card identifies the original NCBI Gene descriptions and the embedding model. Users obtain the two gene-text files directly from the originating repository and use them under the original providers' terms. The public release contains the model's fitted principal-component transform and normalisation parameters. Expression matrices, individual cell records and third-party embedding arrays remain available through the original providers.

- [Replogle genome-scale Perturb-seq](https://doi.org/10.1016/j.cell.2022.05.013) and [Nadig et al.](https://doi.org/10.1038/s41588-025-02169-3)
- [Jiang signalling screens](https://doi.org/10.1038/s41556-025-01622-z)
- [Xaira Therapeutics X-Atlas/Orion dataset and terms](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)
- [Arc Institute Virtual Cell Challenge H1 data](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge)
- [Zhu primary T-cell screens, GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)
- [PIE gene descriptions and source provenance at the pinned revision](https://huggingface.co/datasets/arcinstitute/PIE_sources/tree/fb624a5158a7f37dfa13a015cdacadb985549d26/gene_text)
- [Ensembl stable gene identifiers and data terms](https://www.ensembl.org/info/about/legal/disclaimer.html)

Use the [repository](https://github.com/Qiu-Shizheng/scPertEcho) to contact the maintainers about redistribution or commercial licensing of the toolkit and fitted weights.
