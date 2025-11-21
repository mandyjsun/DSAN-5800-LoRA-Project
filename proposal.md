# 1. Team

**Project title:** Training a lightweight model to pass a Google Interview  
**Team members:** Mandy Sun (ms4821), Sean Morris (spm122)  
**Preferred track:** (B) LoRA finetuning  

---

# 2. Problem Statement & Motivation

## Task
For our project, we want to take a Mistral 7B, a high-performance open source large language model by Mistral AI. Our goal is to train the model to translate natural language instructions (e.g. Determine if this word is a palindrome) into correct executable python code. Beyond merely producing runnable code, we aim for the model to generate high-quality solutions, in that it follows Python conventions, uses clear function structure, includes helpful comments, and demonstrates reasonable run-time.

## Why it matters
High-quality code assistants can greatly improve development speed for experienced engineers, while also reducing the barrier to entry for people less familiar with programming. However, most code assistants fall behind paywalls, or are limited to only those who have access to expensive cloud resources. By exploring how far we can push a smaller, open-sourced model with targeted code datasets, we hope to deliver a computationally lightweight and open-sourced code assistant that is both easy to startup, and effective at everyday programming tasks..

## Desired outcome
For this project, we define ‘success’ as having achieved two key milestones. Of course, our main goal is to deliver a LoRA-finetuned configuration of Mistral 7B that excels in standard benchmarks, as well as a set of tests defined by ourselves that are specifically geared towards our model’s configuration. For the custom evaluation, we intend on serving the model a series of unit tests for each set of output functions, with the goal of distilling performance down into a ‘unit-test pass rate’.  For benchmarking, we elect to use Google’s Mostly Basic Programming Problems (MBPP), consisting of around 1000 beginner to intermediate level programming questions written in python. To accompany our first milestone, we also set out to build a simple interactive demo to illustrate our model’s performance across each test we define. The format of this demo will likely consist of a jupyter notebook that includes model inference runs, performance metrics for our two sets of experiments, and an analysis of different LoRA settings and the impact it has on code quality and success. 

Two milestones define success:

1. A **LoRA-finetuned Mistral 7B** that performs well on MBPP and our custom unit-test suite.  
2. A lightweight **demo notebook** containing:
   - inference examples  
   - performance metrics  
   - comparisons across LoRA ranks  
   - analysis of code quality  

**Benchmarks:**  
- Unit-test pass rate  
- MBPP (~1,000 Python tasks)  

---

# 3. Datasets

## Training Dataset
**Name:** CodeAlpaca-20k  
**Source:** https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k  
**Size:** ~20,000 rows  
**Language:** Python  

## Evaluation Dataset
**Name:** MBPP  
**Source:** https://github.com/google-research/google-research/tree/master/mbpp  (in `mbpp.jsonl`)

**Size:** ~1,000 tasks  
**Language:** Python  

---

# Preprocessing Steps

1. **Python-only filtering**  
   Remove rows with non-Python or malformed code.

2. **Normalize schema**  
   Convert all examples into a consistent structure with:
   - instruction  
   - (optional) input  
   - output python code  

```json
{
 “instruction”: “<natural language task>”
“Input”: “”, # could be unused, could be a list like  [1,2,3,4] 
“Output”: “<python code>”
} 

```

3. **Code cleaning**  
   - normalize indentation to 4 spaces  

4. **Tokenization**  
   - use Mistral tokenizer  

## Train/Val/Test Split
- **80%** train  
- **10%** validation  
- **10%** test (MBPP)  

---

# 4. Baseline

**Baseline model:** Unmodified Mistral 7B (zero-shot)

## Baseline Metrics
- **Functional correctness:** % of tasks where generated code passes unit tests  
- **Syntactic correctness:** % of outputs that run without syntax errors  
- **Human rating:** clarity, readability, adherence to instructions  

---

# 5. Approach (Beyond Baseline)

## LoRA Finetuning
- Finetune on CodeAlpaca-20k  
- Compare LoRA ranks (e.g., 8 vs 32)  
- Evaluate effect on correctness / generalization  

## Data & Prompt Experiments
Compare:
- instruction → code  
- instruction → code + explanation  

**Hypothesis:** explanation-augmented data improves instruction-following but may increase verbosity.

## (Optional) Ablations
- Which task types improve most (algorithms, strings, file I/O)?  
- Error type analysis (off-by-one, missing edge cases, malformed loops, etc.)  

---

# 6. Compute & Resources

- **Jetstream2:** yes/no (TBD)  
- **Model size:** 4–12 GB depending on quantization  
- **Other resources:**  
  - Ollama  
  - Hugging Face  
  - RTX 4070 12GB  
  - Google Colab  

---

# 7. Risks & Scope

## Risks
- **Computation limitations**: Mistral could be difficult to obtain if JetStream or Google Colab provides limited GPU memory
- **Manual evaluation**: qualitative evaluation is very subjective. Manual evaluation could also be incredibly time consuming given the time frame for this project.
- **Overfitting**: We could overfit to the custom unit-test suite and evaluation dataset, but we plan on using a validation set to monitor performance.

## Plan B
- We can limit our scope by reducing dataset size, lowering the sequence length for instructions, or focusing on a smaller benchmark subset from MBPP for evaluation. 
- Reduce sequence length  
- Use only part of MBPP  
- If Mistral7B is too computationally expensive, we might have to switch to a smaller model like CodeLlama-3b.

---

# 8. Milestones

### End of Week 1
- Proposal completed  
- Dataset selection finalized  
- Preliminary cleaning + setup  

### End of Week 2
- LoRA training pipeline implemented  
- First experiments run  
- Collect early metrics  

### End of Week 3
- Final experiments  
- Error analysis  
- Build interactive demo  
- Final report + presentation  
