# Dataset

## Columns Description

You may find the processed datasets under the `splits` folder. These datasets are already cleaned and processed into the same format, saved into CSVs with the following columns:

* corpus [str] : corpus name
* doc_id [str] : document name
* sent_id [int] : sentence id
* eg_id [int] : each sentence can have multiple relations/examples, this indicates the example id count
* index [str] : example unique id 
* text [str] : example input text 
* text_w_pairs [str] : target marked text that includes (`<ARG0>,<ARG1>,<SIGX>`) annotations
* seq_label [int] : target causal label (1 for Causal, 0 for Not Causal)
* pair_label [int] : target causal label (1 for Causal, 0 for Not Causal)
* context [str] : not relevant for CNC that works on single-sentences, to be used for non-consecutive sentence pairs
* num_sents [int] : number of sentences in text column

## Guidelines per Task

### (I) Sequence Classification:
Input column: "text", target column: "seq_label".
Tips: You will need to de-duplicate the dataset by taking only the first "eg_id" (=0) as the main row. This is equivalent to doing a group by with "corpus, doc_id, sent_id" columns. 

### (III) Pair Classification:
Input column: "text_w_pairs", target_column: "pair_label".

### (II) Span Detection
Use datasets under the `grouped` folder instead. Since we are working with examples that have multiple causal relations per input text, we had to group the data such that unique texts have a single row instead of separate indexes for each relation. Namely, we group the data by "corpus, doc_id, sent_id" and keep the first "eg_id" (=0) as the main row. We then create two additional columns to reflect multiple causal relations, if available:
* causal_text_w_pairs [list] : list of up to three causal target marked text that includes (`<ARG0>,<ARG1>,<SIGX>`) annotations. if no causal relation exists, an empty list is returned.
* num_rs [int] : length of list in causal_text_w_pairs


## Paid Data
PDTB and portions of BECAUSE are paid resources. Therefore, we are unable to release them on Github even though we used them extensively for experiments. If you are interested to work with the full corpus, please send us an email and we can either send you the processing scripts or full processed data once we verify you do have the original paid resource access. Thank you for your understanding.
