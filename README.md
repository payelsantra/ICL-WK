# ICL-WK
This repository contains code for the paper "The absence of evidence is not the evidence of absence": Fact Verification via Information Retrieval-based In-Context Learning.

## Prepare Dataset
Run `preprocessing_data.py` to generate the dataset for training and testing.

## Prepare Data from Wikipedia external corpus
To run this file download the `fever.tsv` file from this link `https://doi.org/10.5281/zenodo.8137782` and then run this following command <br> `python3 Wiki_data_preprocessing.py`
## Setup
Install Package Dependencies

`git clone https://github.com/thunlp/OpenPrompt.git`  <br>
`cd OpenPrompt` <br>
`pip install -r requirements.txt`  <br>
Modify the code using `python setup.py develop` <br>
Then `openprompt/plms/__init__.py` have to be modified by our `__init__.py` and kept inside the same place.
For running the experiment run `few-shot.py` and `zero-shot.py` in the ICL-WK directory.


## Zero-shot experiment
run <br>
`python3 zero-shot.py --dataset_path data.csv \` <br>
`--output_prob_path probability_scores.csv \` <br>
`--output_result_path result.csv \` <br>
`--report_file_path 0_shot_report.txt \` <br>
`--prediction_fl_pth pred_fl.pkl \` <br> 
`--temp_fl_pth texts.pkl`


## Few-shot experiment
run <br>
`python3 few-shot.py --dataset_path data_few.csv \` <br>
`--output_prob_path probability_scores_fw.csv \` <br>
`--output_result_path result_fw.csv \` <br>
`--report_file_path 1_shot_report.txt \` <br>
`--prediction_fl_pth pred_1_fl.pkl \` <br> 
`--temp_fl_pth texts_few.pkl`
