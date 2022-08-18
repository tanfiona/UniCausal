# UniCausal
The repository contains the datasets and codes for our work titled "Unified benchmark and model for causal text mining". 

Causality is an important part of human cognition. It is easy for humans to pick up explicit and implicit relations from conversations and text. Causal text mining annotation efforts have been sparse or inconsistent. We proposed UniCausal, a unified benchmark and model for causal text mining across six popular causal datasets and three common tasks. The six datasets reflect a variety of sentence lengths, linguistic constructions, argument types, and more.

Datasets included:
1. AltLex \citep{hidey-mckeown-2016-identifying}
2. BECAUSE 2.0 \citep{dunietz-etal-2017-corpus}
3. CausalTimeBank (CTB) \citep{mirza-etal-2014-annotating, mirza-tonelli-2014-analysis}
4. EventStoryLine V1.0 (ESL) \citep{caselli-vossen-2017-event}
5. Penn Discourse Treebank V3.0 (PDTB) \citep{webber2019penn}
6. SemEval 2010 Task 8 (SemEval) \citep{hendrickx-etal-2010-semeval}.

Tasks covered:
1. Sequence Classification
2. Cause-Effect Span Detection
3. Pair Classification

# Code
#### UniCausal Model
* `run.py`: Performs joint training and testing across three causal text mining tasks

#### Individual Baselines
* `run_seqbase.py`: Sequence Classification
* `run_tokbase.py`: Token Classification a.k.a. Cause-Effect Span Detection
* `run_pairbase.py`: Pair Classification

# Cite Us
```
@misc{tan-etal-2022-unicausal,
  author = {Tan, Fiona Anting and Zuo, Xinyu},
  title = {The Causal News Corpus: Annotating Causal Relations in Event Sentences from News },
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tanfiona/CausalNewsCorpus}}
}
```