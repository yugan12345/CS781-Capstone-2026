# CS781 Capstone 2026 — In-House Legal Intelligence System

A legal language foundation model trained from scratch, and six downstream legal-NLP
systems built on top of it. No pretrained checkpoints, no HuggingFace or fairseq weights,
no external legal data, and nothing that leaves the machine it runs on. Everything here
trains and runs inside a free-tier Colab or Kaggle session.

**Author:** Yugan Jain (231202) · Indian Institute of Technology Kanpur

---

## The problem this is built for

A law firm cannot send privileged client material to a third-party language model API.
That rules out the easy solution and leaves one option: a compact model, trained on the
firm's own corpus, running entirely on infrastructure the firm controls.

The question is then whether a model small enough to self-host can actually support the
work — locating entities in a document, recognising what role each sentence plays in an
argument, predicting how a case is likely to come out, identifying the statutes at issue,
retrieving a relevant prior case, and compressing a judgment into something readable
before a meeting. This repository is seven attempts to answer that, one per capability.

## What the constraints forced

Three constraints shaped every design decision, and it is worth stating them up front
because most of the choices below only make sense in their light.

**No pretrained weights.** The foundation model is trained from scratch on the course
corpus, inside a 50M–300M parameter band. There is no BERT, no LegalBERT, no T5 to fall
back on, so the quality of every downstream system is bounded by a model pretrained with
hours of free GPU time rather than weeks of cluster time.

**A 512-token context.** Legal judgments are long. Measured over the official splits, the
median document runs from roughly 490 tokens in the statute-identification data to over
5,200 in the retrieval data, with 90th percentiles in the thousands and tails past 90,000.
A single 512-token window therefore sees anywhere between a few percent and all of a
document depending on the phase. This is why windowing, chunking and aggregation
experiments appear in four of the seven phases, and why they matter more in some than
others.

**A free session that can disappear.** Sessions are preempted without warning, so every
long run checkpoints on a fixed step interval and every run is written to resume. Memory
discipline is in the architecture rather than bolted on afterwards — see
[Fitting a free session](#fitting-a-free-session).

---

## The tokenizer

A byte-pair-encoding vocabulary of 40,000 tokens, trained on the course corpus
(`phase-1/bpe_40k.json`, loaded by `phase-1/a1_tokenizer.py`). On legal text it runs at
about 1.13 tokens per word.

One measurement drove a downstream decision. The anonymisation placeholders in the data —
`<SECTION>`, `<ACT>`, `<ENTITY>` — cost four or five tokens each, because the vocabulary
has no reason to contain them. Collapsing each to a single rare character recovers roughly
18% of the token budget, so the same 512-token window covers about 1.2× more text. Phase 5
trains the uncollapsed variant as a control, so this is a measurement rather than an
assumption.

---

## The foundation model

Four architectures were pretrained rather than one, because the downstream tasks want
different things. Token classification and retrieval want bidirectional context,
free-form continuation wants a causal decoder, and summarization wants cross-attention.
Training all four means every later phase inherits a measured choice instead of a guess.

All four share the same vocabulary, rotary position embeddings, RMSNorm pre-normalisation,
SwiGLU feed-forward blocks, tied input and output embeddings, and a 512-token context.

| Variant | Architecture | Params | Downstream use |
|---|---|---|---|
| Causal decoder, sequential block | 768d, 19 layers, GQA 12 query / 4 KV heads, `d_ff` 2048 | 150.3M | Phases 2, 3, and the Phase 4 LoRA arm |
| Causal decoder, parallel block | identical dimensions, PaLM-style parallel attention and FFN | ~150M | architecture comparison |
| Masked encoder | 768d, 14 layers, full bidirectional MHA, 12 heads, vocab 40,001 | 130.4M | Phases 4, 5, 7 |
| Encoder–decoder | 640d, 5+5 layers, GQA 10 query / 5 KV heads, `d_ff` 1728 | 77.2M | Phase 6 |

**Grouped-query attention.** Three query heads share each key-value head in the decoder,
which cuts the KV cache by two-thirds at generation time. That is what makes generating
128-word continuations for a thousand test prompts affordable in one session. Tied
embeddings remove a 40,000 × 768 matrix from the optimiser state, which is the difference
between fitting and not fitting on a 16 GB card.

**The parallel-block comparison.** The two decoders differ in exactly one thing at
identical parameter count: the sequential block computes `x + attn(norm₁(x))` and then
`x + ffn(norm₂(x))`, while the parallel block computes attention and feed-forward on a
single shared pre-norm and adds both to the residual at once. Any difference in
convergence is therefore attributable to the topology and nothing else. The parallel
variant also runs under `torch.compile`, which the sequential one does not.

**The encoder's masking.** Masked-language-model pretraining uses document-preserving
token blocks rather than arbitrary 512-token slices, so reconstruction draws on
surrounding legal context rather than local collocation. `[MASK]` is appended as a
model-only id immediately after the tokenizer vocabulary, which is why the encoder's
vocabulary is 40,001 and the decoders' is 40,000 — a detail that matters when loading
either backbone into a downstream head.

**The encoder–decoder.** Rotary embeddings are applied to self-attention only, not to
cross-attention, since relative position between a source and a target sequence is not
meaningful. It is the smallest of the four models because generation, not training, is
the expensive part of Phase 6.

**Corpus mixing.** The corpus is unbalanced — High Courts at roughly 55% of cleaned
tokens, tribunals 34%, Supreme Court 7%, Acts 4%. Sampling in natural proportion
underfits the Supreme Court; sampling uniformly overfits the Acts. Both decoders use
temperature-scaled sampling at α = 0.5, the square-root smoothing of Conneau et al.
(2020), which sits between the two.

**Optimisation.** AdamW with β = (0.9, 0.95) and weight decay 0.1, cosine decay from 3e-4
to 3e-5 after warmup, gradient clipping at 1.0, automatic mixed precision throughout, and
an effective batch of 32 sequences held constant by gradient accumulation.

---

## Adaptation techniques, written from scratch

Phases 4 through 7 implement every adaptation method directly, with no `peft` and no
`bitsandbytes`. Each wraps an `nn.Linear` and reports its own trainable parameter count,
so retention can be read against cost.

| Module | What it computes | Trainable per wrapped layer |
|---|---|---|
| `LoRALinear` | `h = W₀x + (α/r)·BAx` | `r(d_in + d_out)` |
| `LoRMALinear` (LoRMA₊) | `h = ((α/r)·BA + I)·W₀x` — multiplicative, not additive | `2r·d_out` |
| `BottleneckAdapter` | `h' = h + W_up GELU(W_down h + b_down) + b_up` | `2·d·b + b + d` |
| `SoftPrompt` | prepends `m` learned vectors to the embeddings | `m·d` |
| `NF4Linear` | frozen `W₀` stored in 4 bits, dequantized in the forward pass | 0 |
| IA³ | three learned element-wise rescaling vectors per block | 3 vectors per block |
| AdaLoRA | triplet parameterisation with a cubic rank schedule | varies by schedule |

Alongside these: full fine-tuning, partial fine-tuning by layer freezing, and knowledge
distillation.

Two scope notes, stated here rather than buried. The AdaLoRA implementation covers the
triplet parameterisation, the smoothed sensitivity score, the cubic rank schedule and the
orthogonality penalty on `P` and `Q`; it does **not** implement global budget
redistribution across modules, so rank is allocated within each wrapped layer rather than
traded between them. And Phase 3 uses the `peft` library for its LoRA and DoRA arms while
Phase 2 uses no adaptation library at all — both predate the from-scratch implementations
in later phases, and the IA³ arm in Phase 3 is hand-written.

---

## The phases

### Phase 1 — Foundation model pretraining

Four architectures, one tokenizer, one data budget. Each notebook is self-contained from
corpus download through cleaning, tokenization, training, checkpointing and generation.
Cleaning strips markup, URLs, HTML entities, control characters and exotic whitespace, and
discards documents that lose more than a fixed fraction of their characters to it.

`phase-1/`
: `phase1-decoder.ipynb` (sequential causal decoder) ·
  `phase1_palm.ipynb` (parallel block) ·
  `phase1_encoder.ipynb` (masked encoder) ·
  `phase1_encdec.ipynb` (encoder–decoder) ·
  `a1_tokenizer.py`, `bpe_40k.json`

### Phase 2 — Legal named entity recognition

Twelve entity types, scored on strict character offsets. Strict offsets mean a span whose
boundary is one character off counts as a miss, which turns this into a boundary problem
wearing a classification problem's clothes. Three design responses follow from that.

An **offset-tracking encoder** carries every subword back to its exact character span in
the original document, so predictions can be scored where the grader scores them. A
**BiLSTM-CRF head** decodes the tag sequence globally rather than per token, because a
token's correct tag depends on its neighbours — an `I-` tag is only legal after a matching
`B-`. And a **span-merge rule** reconstructs entities from tags, breaking at a change of
entity type or at any token with no entity prediction, then trimming trailing punctuation
and brackets.

The experiments vary the label scheme (BIO prefixes against flat tags, 25 classes against
13), the tagging unit (subword pieces against pooled whole words), the backbone (decoder
against encoder), and whether the head is trained with knowledge distillation from the
stronger backbone. A separate sanity notebook checks the backbone before any training
runs: that hidden states have not collapsed, that different tokens receive different
representations, and that gradients reach the backbone through the new head.

`phase-2/`
: `exp1_full_ft_bio_fixed.ipynb` · `exp1_partial.ipynb` ·
  `exp1_full_fine-tuning-no-bio-tag.ipynb` · `phase2_word_vs_subword_bio_(1).ipynb` ·
  `exp_encoder_bio.ipynb` · `exp_kd_bio.ipynb` · `test_decoder_for_ner.ipynb`

### Phase 3 — Rhetorical role prediction

Thirteen role labels over the sentences of a judgment, from 80 labelled documents. That
is the smallest supervised set in the project by three orders of magnitude, and the
experiment design is built around the scarcity rather than against it.

The arms are ordered by **how far the backbone is allowed to move** — frozen first, then
parameter-efficient adaptation, then joint training — because with 80 documents the right
question is not which method wins but whether the backbone needs to move at all. A frozen
run also answers what the Phase 1 representation already contains, which is the most
useful single thing to know before spending compute.

Every arm sits under a hierarchical head: each sentence is encoded independently, then a
BiLSTM reads the sequence of sentence vectors so that a sentence's role can depend on its
neighbours, and a CRF decodes the label sequence. Judgments follow a near-canonical
order — facts, then arguments, then precedent, then ruling — and that order is signal the
CRF can use.

Two measurements make this phase's conclusions checkable. A **three-seed replication**
establishes how much of any difference between arms is noise, which sets a threshold below
which differences should not be interpreted. And the distillation teacher is built from
**out-of-fold targets**: five fold models, each producing soft labels only for documents
it never saw, so no training signal comes from a model that has memorised the document it
is labelling. The soft targets are the CRF's posterior marginals, conditioned on the whole
document, rather than per-token emission softmaxes — the structure is part of what gets
distilled.

`phase-3/`
: `rhetorical_role_encoder_hierarchical_crf.ipynb` · `rhetorical_role_ia3_crf.ipynb` ·
  `rhetorical_role_dora_bilstm_crf.ipynb` · `rhetorical_role_lora.ipynb` ·
  `rhetorical_role_lora_bilstm_v2.ipynb` · `rhetorical_role_distillation_crf.ipynb` ·
  `Phase3_RR_Chunk_Attention_(2).ipynb`

### Phase 4 — Court judgment prediction

Binary prediction over a balanced set of judgments. Six adaptation techniques share one
library — the same data loading, windowing, caching, classifier head, training loop and
evaluation code — so the only thing that differs between notebooks is the adaptation
method. Notebook `01` is the reference the others are read against.

Three diagnostics run in every notebook, and they are the reason the results mean
anything. A **majority-class and random-baseline floor**, so a score is read as a distance
above chance rather than as an absolute. A **confidence interval** on accuracy at the dev
set's size, so differences smaller than the interval are reported as indistinguishable
rather than as wins. And a **threshold sweep**, because a sigmoid emits a score and the
decision rule that turns it into a label is a free parameter that nobody optimised during
training.

The distillation student is a **layer-subset** model: evenly spaced layers drawn from the
teacher, which preserves the depth-wise structure of the function better than narrowing
every layer would. Each dev instance carries five independent expert annotations collapsed
into one binary training label, which makes this phase a natural place to ask whether a
teacher's soft probabilities can restore information the collapse destroyed.

`phase-4/`
: `cjpe_01_full_finetune_(2).ipynb` · `cjpe_02_layer_freezing.ipynb` ·
  `cjpe_03_lora.ipynb` · `cjpe_04_adalora_(1).ipynb` · `cjpe_05_adapters_(1).ipynb` ·
  `cjpe_06_distillation.ipynb`

### Phase 5 — Legal statute identification

Multi-label prediction over 100 distinct statute ids, on the largest supervised set in the
project. The head is 100 independent sigmoids trained with binary cross-entropy, and the
phase splits into eight adaptation experiments over one shared window and one shared head,
plus a ninth notebook of secondary studies.

**Thresholds are treated as a separate object from the model.** A sigmoid gives a score,
not a decision, and a macro-averaged metric over 100 imbalanced labels rewards a different
cut for each one. Every experiment is therefore reported three ways: at a naive 0.5 cut, at
a single tuned global threshold, and at per-label thresholds. Keeping the three separate is
what makes it possible to say how much of a result came from the model and how much from
the decision rule.

This is also the only phase whose median document nearly fits the context window, which
makes it the right place to test whether windowing matters at all. The length distribution
was measured *before* the windowing study was designed, so the study was run with a
prediction attached rather than as an open sweep.

The five secondary studies cover loss functions and class balancing (plain, positive-weighted,
focal and asymmetric losses), long-document windowing (head, tail, head-and-tail),
whether collapsing the anonymisation placeholders is worth the preprocessing, chunked
whole-document inference with mean against max pooling, and whether enforcing the strong
statute co-occurrence implications found in the training set helps at prediction time.

`phase-5/`
: `lsi_01_full_finetune.ipynb` → `lsi_08_distillation (2).ipynb` ·
  `lsi_09_studies_(1) (1).ipynb` · `Phase5_Train_8Chunks_Mean (1).ipynb`

### Phase 6 — Abstractive summarization

Generation on the encoder–decoder backbone. No development split shipped with the data,
so a fixed fraction of the training pairs is held out under a fixed seed and kept disjoint
throughout, which matters because every experiment in the phase is compared on it.

The experiment set covers full fine-tuning, partial fine-tuning with the encoder frozen
(testing whether Phase 1 had already learned an adequate document representation and only
generation remained to be learned), LoRA at all three attention sites, multiplicative
LoRMA₊, and bottleneck adapters in both the Houlsby placement (two adapters per block) and
the Pfeiffer placement (one), which is a clean controlled pair.

Every run reports a **repetition rate** alongside the overlap metrics, and a mean output
length against the reference length. Those two columns turn out to carry more of the story
than the adaptation method does, which is the kind of thing an instrumented run tells you
and a leaderboard does not.

Two honest notes on method. Model selection during training uses development **loss**
rather than a generation metric, because generating summaries for every candidate
checkpoint costs minutes per evaluation and the session budget does not allow it — and the
distillation run shows that loss and overlap can disagree. And every run is capped by wall
clock rather than by convergence, with the learning-rate schedule built to fit the cap.

`phase-6/`
: `E1_full_ft.ipynb` · `E2_decoder_only.ipynb` · `E3_lora.ipynb` · `E4_houlsby.ipynb` ·
  `E5_pfeiffer.ipynb` · `E8_lorma.ipynb` · `E6_distill.ipynb` · `E6b_student_nokd.ipynb`

### Phase 7 — Prior case retrieval

Given a query judgment, return the most relevant prior cases from a candidate pool. Three
things about this phase were settled before any model was trained, and each saved a wasted
run.

**The ceiling was computed first.** The metric caps predictions at seven, so queries with
more than seven relevant prior cases cannot reach full recall. Knowing the attainable
maximum before training is what lets a result be read as a fraction of what is reachable
rather than as a fraction of one.

**A lexical baseline was built first**, both as the number a neural retriever has to beat
and as a source of training negatives. Legal relevance turns heavily on citations, statute
numbers, party names and formulaic phrasing — exactly the surface features a character
n-gram retriever captures well.

**The pipeline order is a memory decision, not a preference.** Late interaction over the
full pool means holding sixteen chunk embeddings per document and comparing every chunk
pair, which is quadratic in pool size and does not fit a free session. Lexical retrieval
narrows the pool first; the neural model only ever scores what survives.

Documents are chunked and scored under three aggregations — the first chunk alone, the
mean of all chunks, and the best-matching chunk pair — all evaluated from a single
encoding pass per epoch, with the best kept.

The phase's most useful finding came from a failure. Mining hard negatives from the
model's own top-ranked candidates collapsed training, because nearly every candidate in
this pool is relevant to *some* query, so the mined negatives were largely unlabelled
positives being pushed away from their own queries. Negatives mined once from the lexical
baseline, skipping the top of its ranking, fixed it. The general lesson is that
hard-negative mining assumes relevance is sparse, and in this corpus it is dense. The
training core documents the collapsed configuration and the corrected one side by side.

`phase-7/`
: `phase7_full_fine_tune.ipynb` · `phase7_lorma.ipynb` · `phase7_ia3.ipynb` ·
  `phase7_distill.ipynb` · `phase7_re_ranker.ipynb`

---

## Fitting a free session

Memory and time discipline is a design constraint here, not an optimisation pass.

**Cache whatever is frozen.** Phase 3's frozen arms encode each sentence once into a cache
of a few tens of megabytes, release the encoder from device memory entirely, and train the
BiLSTM-CRF head on cached vectors at seconds per epoch. That is what made multi-seed
replication affordable at all — and a noise-floor measurement you cannot afford is a
measurement you do not make.

**Sub-batch inside the example.** A Phase 3 document is a few hundred sentences and cannot
be encoded in one pass, so sentences are batched within the document and gradients
accumulated across the sub-batches.

**Probe the batch size at startup.** Rather than hard-coding a batch size per accelerator,
runs attempt a real forward and backward pass and halve until it fits, then hold the
effective batch constant with gradient accumulation.

**Spend the budget on optimiser steps.** An early adapter run accumulated gradients over
four micro-batches, reached only a few hundred optimiser steps in its time budget, and its
adapters barely left their initialisation. Dropping accumulation to one at the same wall
clock quadrupled the number of updates. Under-training a parameter-efficient method looks
identical to the method failing, so step counts are recorded with every such result.

**Survive preemption.** Checkpoints are written on a fixed step interval, runs resume from
the latest one, and chunk caches are written per document so an interrupted encoding pass
continues with the documents it has not done yet.

Throughout: mixed precision with a gradient scaler, PyTorch's fused scaled-dot-product
attention, and `torch.compile` where the model supports it.

---

## Methodology notes

These are the practices the phases share, and they are the part of this repository most
worth reusing.

- **Report a floor.** Majority class, a random baseline, a frozen backbone, or a classical
  non-neural method. A metric with no floor beside it cannot be interpreted.
- **Report a ceiling where the metric imposes one.** Phase 7's cap on the number of
  predictions bounds the attainable score, and that bound belongs next to the result.
- **Measure the noise before interpreting a difference.** Phase 3 replicates across seeds
  and Phase 4 computes a confidence interval at the dev set's size. Differences inside
  those bounds are reported as indistinguishable.
- **Report cost next to quality.** Every adaptation module reports its own trainable
  parameter count, so matching full fine-tuning at a fraction of a percent of the
  parameters is visible as the result it is.
- **Run the control, not just the treatment.** Every distillation experiment has a
  no-teacher sibling — the same student architecture trained on hard labels alone — because
  without it a student's score says nothing about whether the teacher contributed.
- **Measure the data before designing around it.** Length distributions, label supports and
  tokenizer behaviour were measured first, which is how the windowing and placeholder
  studies came to have predictions attached rather than being open sweeps.

---

## Known limitations

Stated plainly, because they bound how the results in these notebooks should be read.

- **Single seed outside Phase 3.** Phases 4 through 7 report one run per arm, while Phase 3
  measured a seed-to-seed spread wider than most between-method differences elsewhere.
  Technique orderings in those phases should be treated as unreplicated.
- **Schedules fixed by epoch count.** Several arms record their best checkpoint on the
  final step of their schedule, which means the run ended because the counter ran out
  rather than because the model converged. The cosine learning-rate decay reaches its
  floor at the same moment, so extending a run means re-specifying the schedule, not just
  adding steps.
- **Selection criterion differs from the generation metric.** Phases 1 and 6 select
  checkpoints on validation loss, and Phase 6's own distillation run demonstrates that loss
  and overlap metrics can disagree.
- **No decode-strategy experiments.** Neither generation phase varies sampling
  temperature, beam width or repetition penalty, despite both reporting a repetition
  diagnostic.
- **Ensembling used in one phase only.** Phase 3 combines its fold models; the trained
  checkpoints in the other phases are never combined.
- **Incomplete committed outputs.** The AdaLoRA and adapter notebooks in Phase 4, the flat-tagging
  notebook in Phase 2, and the parallel-decoder notebook in Phase 1 are committed without
  their run logs.

---

## Setup

The notebooks run on Colab or Kaggle. The dataset is pulled from the official
`Exploration-Lab/CS-781-Capstone` HuggingFace repository, which requires a token.

Store the token as a Colab secret named `HF_TOKEN` — never paste it into a cell:

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN") or None   # an empty string is not None
```

The `or None` matters: an empty string produces `Illegal header value b'Bearer '` rather
than falling back to an anonymous request.

```bash
pip install torch transformers datasets huggingface_hub scikit-learn rouge-score
```

Neither `peft` nor `bitsandbytes` is required for Phases 4 through 7 — those
implementations are in this repository. Phase 3's LoRA and DoRA arms do use `peft`.

Each phase expects the relevant Phase 1 checkpoint plus `bpe_40k.json` and
`a1_tokenizer.py` to be reachable, either in the notebook directory or in Drive; paths are
set in the first configuration cell of every notebook.

---

## Repository layout

```
phase-1/   foundation model pretraining — two decoders, masked encoder,
           encoder–decoder, plus the tokenizer and its vocabulary
phase-2/   legal NER — BIO and flat tagging, word and subword units,
           both backbones, distillation, and a backbone sanity harness
phase-3/   rhetorical roles — frozen / PEFT / joint arms under a BiLSTM-CRF,
           learned chunk attention, out-of-fold distillation
phase-4/   judgment prediction — six adaptation techniques over one shared library
phase-5/   statute identification — eight techniques plus threshold, loss,
           windowing, placeholder, chunking and label-structure studies
phase-6/   summarization — E1–E8, including the Houlsby/Pfeiffer adapter pair
           and a distillation control
phase-7/   prior case retrieval — lexical baseline, bi-encoder arms,
           distillation, cross-encoder reranker
```

---

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Touvron et al. (2023), *LLaMA* — RMSNorm pre-norm, SwiGLU, rotary embeddings
- Chowdhery et al. (2022), *PaLM* — the parallel attention and feed-forward block
- Su et al. (2021), *RoFormer* — rotary position embeddings
- Shazeer (2019), *Fast Transformer Decoding* and Ainslie et al. (2023), *GQA* — grouped-query attention
- Devlin et al. (2019), *BERT* — masked language modelling
- Conneau et al. (2020), *XLM-R* — temperature-scaled corpus sampling
- Hu et al. (2021), *LoRA*; Zhang et al. (2023), *AdaLoRA*; Liu et al. (2022), *IA³*;
  Houlsby et al. (2019) and Pfeiffer et al. (2021), *bottleneck adapters*;
  Dettmers et al. (2023), *QLoRA* — NF4 quantisation
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*
- Lafferty et al. (2001), *Conditional Random Fields*
- Robertson and Zaragoza (2009), *BM25*
- Khattab and Zaharia (2020), *ColBERT* — late interaction and max-similarity aggregation
- Oord et al. (2018), *InfoNCE*
