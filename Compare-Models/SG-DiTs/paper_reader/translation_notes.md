# Translation and extraction notes

- Status: implementation-focused draft reader, not a full 19-page line-by-line translation.
- Source: local selectable-text PDF supplied by the user; no OCR was required for the main text.
- Scope chosen for reproduction: feature definitions, DDPM equations and schedule, DiT/AdaLN architecture, objective, rolling window, and reported hyperparameters.
- The English blocks in `paper.md` are concise source-grounded restatements, not long verbatim quotations. Chinese blocks are faithful technical translations of those statements.
- Figure 10 was cropped from the source PDF and is linked to block S006. The uncropped page image is kept under `diagnostics/` only for audit and is not a manuscript asset.
- Material not represented block-by-block: extended introduction, related-work survey, dataset descriptions, most result tables/figures, conclusion, and references.
- Key disclosure gaps affecting reproduction: SG window/polyorder, entropy binning, MLE estimator, optimizer, epoch count, sample count, exact tensor contract, and variance-head supervision.
- Repository adaptations are documented separately in `../REPRODUCTION.md`; they must not be attributed to the paper authors.
