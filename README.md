# UniCausal
Causality is an important part of human cognition. It is easy for humans to pick up explicit and implicit relations from conversations and text. Causal text mining annotation efforts have been sparse or inconsistent. 

We introduce UniCausal, a unified benchmark and model for causal text mining across six popular causal datasets and three common tasks. 

### Datasets included:
The six datasets reflect a variety of sentence lengths, linguistic constructions, argument types, and more. 

<img align="center" src="assets/Table1_DataDifferences.PNG">

<br>

<img align="center" height=250 src="assets/Table2_DataSizes.PNG">

<br>

### Tasks covered:

(I) Sequence Classification <br>
(II) Cause-Effect Span Detection <br>
(III) Pair Classification <br>

<img align="center" height=250 src="assets/Figure1_Tasks.PNG">

<br>

For more details and analysis, please refer to our corresponding paper titled "UniCausal: Unified benchmark and model for causal text mining".

<br>

# Code

### Individual Baselines

(I) `run_seqbase.py`: Sequence Classification <br>
(II) `run_tokbase.py`: Token Classification a.k.a. Cause-Effect Span Detection <br>
(III) `run_pairbase.py`: Pair Classification <br>

<br>

# Links to Original Datasets
1. [AltLex](https://github.com/chridey/altlex) (Hidey and McKweon, 2016)
2. [BECAUSE 2.0](https://github.com/duncanka/BECAUSE) (Duneitz et al., 2017)
3. [CausalTimeBank (CTB)](https://github.com/paramitamirza/Causal-TimeBank) (Mirza et al., 2014; Mirza and Tonelli, 2014)
4. [EventStoryLine V1.0 (ESL)](https://github.com/tommasoc80/EventStoryLine) \citep{caselli-vossen-2017-event}
5. [Penn Discourse Treebank V3.0 (PDTB)](https://catalog.ldc.upenn.edu/LDC2019T05) (Webber et al., 2019)
6. [SemEval 2010 Task 8 (SemEval)](https://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11) (Hendrickx et al., 2010)

<br>

# Cite Us
If you used our repository or found it helpful in any way, please do cite us in your work:
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

If you have feedback or features/datasets you would like to contribute, please email us at tan.f[at]u.nus.edu.

[Current version: 1.0.0]