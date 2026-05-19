# Task: Zotero Idea-Origin References

Tags: #progress #references #zotero #bibliography #idea-origin

Related notes: [[2026-05-13_zotero_project_references]] [[2026-05-08_proposal_alignment_audit]] [[DATASET_PROJECT_PLAN]] [[EXPERIMENT_REPORT]]

## 1. Goal

The user asked for a deeper pass over the project to find papers that introduced or motivated the ideas used in SetConCA V2, not just explicitly linked references.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-13 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Extra local sources checked | `C:\Users\MPC\Documents\code\SetConCA\conca_math.txt`, `C:\Users\MPC\Documents\code\SetConCA\set_conca_math.txt` |
| Zotero route | Local Zotero connector server at `127.0.0.1:23119` |

## 3. Method

Searched docs, model code, training losses, and proposal notes for method breadcrumbs:

- ConCA / Sparse ConCA / Set-ConCA.
- set aggregation and mean pooling.
- TopK sparse codes, sparse autoencoders, dictionary learning, and monosemanticity.
- InfoNCE and contrastive semantic loss.
- linear bridges, Procrustes, ridge regression, and cross-model latent communication.
- activation intervention and steering vectors.

Then mapped those project mechanisms to the most relevant originating or canonical papers.

## 4. References Added

Added 21 additional Zotero items tagged `idea-origin`:

- Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs.
- Set-ConCA: Extending Concept Component Analysis from Point Representations to Representation Sets.
- Deep Sets.
- Layer Normalization.
- Representation Learning with Contrastive Predictive Coding.
- SimCLR.
- Toy Models of Superposition.
- Towards Monosemanticity.
- Sparse Autoencoders Find Highly Interpretable Features in Language Models.
- Scaling and Evaluating Sparse Autoencoders.
- BatchTopK Sparse Autoencoders.
- Improving Dictionary Learning with Gated Sparse Autoencoders.
- JumpReLU Sparse Autoencoders.
- The Linear Representation Hypothesis and the Geometry of Large Language Models.
- Relative Representations Enable Zero-Shot Latent Space Communication.
- Inference-Time Intervention.
- TCAV.
- K-SVD.
- Olshausen and Field sparse coding.
- Ridge Regression.
- Orthogonal Procrustes.

## 5. Output Files

| File | Purpose |
| --- | --- |
| `docs/references/setconca_idea_origin_references_zotero_audit.json` | Machine-readable audit of the second Zotero save batch. |
| `docs/references/setconca_idea_origin_references.bib` | BibTeX fallback/import file for the idea-origin batch. |
| `docs/references/setconca_project_references.bib` | Appended with the idea-origin references. |

## 6. Result

Zotero returned HTTP `201 Created` for the second save batch.

These references are not all direct dependencies. They are the method lineage: the papers most likely to explain where the architecture, losses, baselines, bridge methods, and steering setup came from.
