# CS781 Capstone 2026 — In-House Legal Intelligence System

A legal language model and six downstream legal-NLP systems, **pretrained entirely from
scratch** on the official course corpus. No pretrained checkpoints, no HuggingFace or
fairseq weights, no external legal data. Everything here was built to train and run inside
a free-tier Colab/Kaggle session.

**Author:** Yugan Jain (231202) · Indian Institute of Technology Kanpur
**Codabench:** `y_231202`

---

## Results

| Phase | Task | Metric | Score | Rank | Subs |
|:-----:|------|--------|:-----:|:----:|:----:|
| 1 | Legal LM pretraining | BERTScore F1 | **0.8299** | 83 | 5 |
| 2 | Legal NER | Strict Macro-F1 | **0.4922** | 85 | 4 |
| 3 | Rhetorical role labelling | Macro-F1 | **0.5094** | 52 | 9 |
| 4 | Court judgment prediction | — | *not scored* | — | 6 |
| 5 | Legal statute identification | Macro-F1 | **0.3696** | 77 | 9 |
| 6 | Abstractive summarization | BERTScore F1 | **0.8339** | 84 | 8 |
| 7 | Prior case retrieval | Micro-F1@7 | **0.4087** | 58 | 7 |

48 valid submissions total. Full analysis in [`report/`](report/).

---

## The foundation model

Four architectures were pretrained and compared, rather than committing to one. All share a
custom 40k legal BPE vocabulary, RoPE, RMSNorm pre-normalisation, SwiGLU blocks, tied
embeddings, and a 512-token context, and all sit inside the 50M–300M parameter cap.

| Model | Params | Val loss | Notes |
|-------|:------:|:--------:|-------|
| Masked encoder | 161.1M | 1.697 | document-level masking |
| PaLM-style decoder | 181.0M | **2.232** | parallel attention/FFN block |
| LLaMA-style decoder | 181.0M | 2.281 | sequential block |
| Encoder–decoder | 128.4M | — | GQA 12:4, for Phase 6 |

At equal parameter count the parallel block reached a lower loss than the sequential one,
so the topology was an efficiency gain rather than a quality trade. Grouped-query attention
at 3:1 cuts the KV cache by two-thirds; tied embeddings remove a vocabulary-sized matrix
from the optimiser state.

---

## Adaptation techniques

Nine techniques, **written from scratch for Phases 4–7** — no `peft`, no `bitsandbytes`:

full fine-tuning · partial fine-tuning · LoRA · LoRMA₊ (multiplicative) · AdaLoRA · IA³ ·
bottleneck adapters (Houlsby and Pfeiffer) · QLoRA over a from-scratch NF4 base ·
prompt tuning · knowledge distillation

> Phase 3 uses the `peft` library for LoRA and DoRA, and Phase 2 uses no PEFT at all —
> both predate the course's treatment of from-scratch implementations.

---

## What the data forced

Measured over the official splits, before any modelling decisions:

| Phase | Records | Median tokens | p90 | >512 tok | kept @512 |
|:-----:|--------:|--------------:|----:|---------:|----------:|
| 1 | 18,826+ | 3,034 | 14,756 | 89% | 6.6% |
| 2 | folds | 5,411 | 10,626 | 97% | 7.7% |
| 3 | 80 | 4,739 | 24,782 | 100% | 5.3% |
| 4 | 37,387 | 2,961 | 9,304 | 92% | 9.9% |
| 5 | 42,750 | **634** | 4,076 | **56%** | **23.7%** |
| 6 | 7,030 | 3,994 | 9,721 | 100% | 9.1% |
| 7 | 827 | 5,225 | 14,618 | 99% | 6.4% |

A single 512-token window sees between 5% and 24% of a document. Supervision varies by
three orders of magnitude — 80 labelled documents in Phase 3 against 42,750 records in
Phase 5 — so the same adaptation technique is not being asked the same question in each
phase. Phase 5 is the only phase whose median document nearly fits the context, which is
why its windowing study came out near-vacuous.

---

## Per-phase approach and findings

**Phase 1 — foundation model.** Four architectures compared head to head. The encoder uses
document-level masking so reconstruction draws on surrounding legal context rather than
local collocation. Checkpoints every 1000 steps, because a free session is preempted
without warning and a run that cannot resume cannot finish.

**Phase 2 — NER.** Strict offset scoring makes boundary placement the whole problem. Tried
BIO tagging ±CRF ±BiLSTM and all four combinations, then flat tagging to shrink the
prediction space (25 classes → 13 across twelve entity types), recovering spans afterwards
with a merge rule that breaks at a change of type *or* at any token with no entity
prediction. BIO full fine-tuning won at **0.4813** dev; flat tagging reached 0.2694. The
0.212 gap is what the boundary tag is worth on this data.

**Phase 3 — rhetorical roles.** Only 80 labelled documents, so the arms were ordered by how
far the backbone has to move: frozen → PEFT → full fine-tuning, each under a BiLSTM with and
without a CRF. Frozen decoder + BiLSTM-CRF won at **0.4592**. A three-seed baseline measured
the noise floor at ±0.017, setting a significance bar of 0.039 — against which IA³'s +0.0094
over frozen is **not a detectable difference**, despite training only 48,640 parameters
(0.03% of the backbone).

**Phase 4 — judgment prediction.** Distillation from an evenly-spaced layer-subset student
**beat its own teacher**, 0.7052 against 0.6908, with PEFT methods retaining 88–97% of full
fine-tuning. Each instance carries five independent expert annotations collapsed into one
training label, which is the likely explanation.

**Phase 5 — statute identification.** Distillation won again (**0.3615** vs 0.3546 full FT,
and vs a 0.3456 no-teacher control). Frozen floor 0.2127 makes adaptation worth 0.142.
Separating the decision threshold from the model was worth ~0.06 macro-F1 — comparable to
the entire spread between techniques.

**Phase 6 — summarization.** Tried full FT, a frozen encoder (on the hypothesis that Phase 1
had already learned an adequate document representation and only generation remained), LoRA,
LoRMA₊, and adapters in both placements. Head+tail windowing beat head-only
(**2.8885** loss / 0.2150 ROUGE-L). The dominant failure is repetition, not adaptation:
63–79% of generated tokens repeat earlier ones across every variant, a 0.165 spread against
a 0.023 ROUGE-L spread between techniques. Distillation **diverged** here — the one phase
where it failed.

**Phase 7 — prior case retrieval.** A two-stage pipeline: a 2–3-gram TF-IDF stage retrieves
candidates and a cross-encoder reranks them (**0.4202** dev / 0.4087 test). The lexical stage
goes first for a memory reason — late interaction over the full pool means holding sixteen
chunk embeddings per document and comparing all chunk pairs, which is quadratic and does not
fit. Negative mining from the model's own top-ranked candidates collapsed training, because
nearly every candidate is relevant to some query and those negatives were largely unlabelled
positives; BM25-mined negatives skipping the top of the ranking fixed it.

---

## Fitting a free session

Memory discipline is in the architecture, not bolted on afterwards:

- **Cache whatever is frozen.** Phase 3's frozen arms encode each sentence once into a cache
  of a few tens of megabytes, release the encoder from device memory, and train the
  BiLSTM-CRF head on cached vectors at seconds per epoch — which is what made the multi-seed
  noise-floor measurement affordable at all.
- **Sub-batch inside the example.** A Phase-3 document is a few hundred sentences and cannot
  be encoded in one pass.
- **Probe the batch size at startup**, halving until a real forward and backward pass fits,
  then hold the effective batch constant with gradient accumulation.
- **Spend the budget on optimiser steps.** An early adapter run accumulated over four
  micro-batches, reached only a few hundred steps, and its adapters barely left their
  initialisation — accumulation of one at the same wall clock quadrupled the updates. Under-
  training a PEFT method looks identical to the method failing, so step counts are recorded
  with every PEFT result.
- AMP + GradScaler throughout, fused SDPA, `torch.compile` on the PaLM variant, and
  checkpoint-every-1000-steps for preemption survival.

---

## Setup

Notebooks run on Colab or Kaggle. The dataset is pulled from the official
`Exploration-Lab/CS-781-Capstone` HuggingFace repo, which requires a token.

**Store your token as a Colab secret named `HF_TOKEN`** (🔑 in the left sidebar) — never
paste it into a cell:

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN") or None   # empty string != None
```

An empty string produces `Illegal header value b'Bearer '` rather than an anonymous
request, so the `or None` matters.

```bash
pip install torch transformers datasets huggingface_hub scikit-learn rouge-score
```

No `peft` or `bitsandbytes` is required for Phases 4–7 — those implementations are in this
repo.

---

## Repository layout

```
phase1/     foundation model pretraining (encoder, decoders, encoder–decoder)
phase2/     NER — BIO and flat tagging variants
phase3/     rhetorical roles — frozen / PEFT / full FT × BiLSTM × CRF
phase4/     judgment prediction — incl. distillation
phase5/     statute identification — incl. threshold and window studies
phase6/     summarization — E1–E8 experiment series
phase7/     prior case retrieval — lexical, bi-encoder, reranker
analysis/   per-phase lookup notebooks + p0_data_stats.ipynb
report/     CS781-CP-Yugan-Jain-231202-report.pdf + LaTeX source
```

*(Adjust to match your actual folder names as you upload.)*

---

## Report

[`report/CS781-CP-Yugan-Jain-231202-report.pdf`](report/) — full methodology, ablations,
error analysis and LLM usage declaration.
