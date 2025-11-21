# ANLY-5800 Final Project – Proposal Template

Use this as a guide for your **2-page max** project proposal (one per group). You can write it directly in this file or a separate PDF in your repo.

---

## 1. Team

- **Project title**:
- **Team members**: Names & NetIDs
- **Preferred track**: (A) Tiny LM, (B) LoRA finetuning, (C) Agent, (D) Analysis, (E) Student-defined

---

## 2. Problem statement & motivation

- **Task**: What are you trying to do? (e.g., sentiment classification, QA, code generation, tool-using agent)
- **Why it matters**: Briefly explain why this problem is interesting or important (scientifically or practically).
- **Desired outcome**: What will success look like in 3 weeks?

---

## 3. Datasets

- **Primary dataset(s)**: Name, source (link), size (# examples, languages, domains)
- **Preprocessing**: What cleaning/tokenization/formatting is needed?
- **Train/val/test split**: How will you split or use existing splits?

---

## 4. Baseline

- **Baseline model/system**: What is the simplest reasonable model you will implement in Week 1?
  - Examples: TF-IDF + logistic regression, zero-shot LLM, off-the-shelf checkpoint without finetuning, a tiny RNN.
- **Baseline metrics**: What metric(s) will you report (accuracy, F1, BLEU/ROUGE, perplexity, etc.)?

---

## 5. Approach (beyond baseline)

Describe your **core idea(s)** for improving over the baseline, tied to the course content.

Examples:

- Track A: Modify Transformer depth/width, context length, or data size and study effects.
- Track B: LoRA finetuning with different ranks, target modules, or objectives.
- Track C: Tool-using agent with ReAct-style reasoning, or multiple tools.
- Track D: Scaling study, robustness analysis, or comparison of finetuning strategies.
- Track E: Any combination of the above, but clearly grounded in course topics.

You don’t need every detail now, but you should have **at least two concrete improvements or experiments** planned beyond the baseline.

---

## 6. Compute & resources

- **Will you use Jetstream2?** (yes/no)
- **Rough plan**: Expected model sizes, batch sizes, and approximate training time.
- **Other resources** (if any): local GPUs, other cloud providers, external APIs.

---

## 7. Risks & scope

- **What could go wrong?** (e.g., data too noisy, model too big to train, evaluation too hard)
- **Plan B**: If your original idea is too ambitious, what scaled-down version will you execute?

---

## 8. Milestones

Very briefly, list what you plan to achieve by:

- **End of Week 1**:
- **End of Week 2**:
- **End of Week 3**:

These should align with the course-wide milestones in `project/README.md`.