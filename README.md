# Testing4LMaaS
## TL;DR
CRATE: A unified contrastive testing framework for Embedding API Service.

## Repo Structure
1. `data` : It contains the dataset we used and the test suite we generated in our experiment.
2. `Testing.py` : It contains the source codes of the validation stage in CRATE.
3. `Evaluation.py` : It contains the source codes about how to construct the ground truth and evalute the test results by such as precsions.
4. `TrainDCs.py` : It contains the source codes about how to train 14 downstream  classifiers.
5. `Fixing.py` : It contains the source codes of our fixing experiment.

```
-Generator/
    -SentencePairTrans.py
    -SentenceTrans.py
-data/
    -builder.py
    -ctset1.py
    -SA.json
    -sst_train.json
-DClassifier
    -__init__.py
-buildContrastset.py
-Testing.py
-TrainDCs.py
-Evaluation.py
-Fixing.py
-plms.txt
-mutatePLMs.py
-README.md
-requirements.txt
```
## Setup

CRATE requires specific versions of some packages, which may conflict with others. Thus, a virtual environment is recommended.
Firstly, you need to install python 3.8 and PyTorch 1.10.2 from [here](https://pytorch.org/get-started/locally/). Then install other packages, run:
```
pip install -r ./requirements.txt
```
What's more, if you want to run the generator module, please install [this third-party library](https://github.com/GEM-benchmark/NL-Augmenter) manually.


## Usage
Here, we give a demo to show how to use our codes.

To test the LLMs:
```
python Testing.py --contrastset ./data/ctset1.json --plm bert-base-uncased --gpu 1 --output ./data/results/bert.json
```

To get the ground truth and evaluate the test results:
```
python TrainDCs.py --dataset ./data/sst_train.json --plm bert-base-uncased --gpu 1 --output_dir ./data/DCs/

python Evaluation.py --bugs ./data/results/bert.json --contrastset ./data/ctset1.json --plm bert-base-uncased --dclf_dir ./data/DCs/ --gpu 1 --results ./data/results/eval.json
```

To fix the LLM by test results:
```
python Fixing.py --bugs ./data/results/bert.json --contrastset ./data/ctset1.json --plm bert-base-uncased --gpu 1 --output ./data/fixmodel
```

The optional Args are:

|Argument | Help | Default |
|----------|:-------------:|------:|
|--cache | the cache dir where the pretrained LLMs from Hugging face are saved locally | -|
|--customodel | name of custom model , which can not be downloaded directly from Hugging face |  - | 
|--customcache | path of the custom model. if the argument is used , the plm and cache argument is unavaliable | - |
|--dis | the distance metric | ['l2','l1','cos'] |
|--thres | the threshold | ['zero','min','m1s','m2s'] |
|--epoch | the training epochs of classifiers | 100 |


We provide the test suite used by our evluation at `./data/ctest1.json`, if you want to generate your own test suite, you can do like following:
```
python buildContrastset --trans ./data/SA.json --input  ./data/sst_train.json --output XXX.json
```

## Reproducing Tips:
We list the LLMs we employed in our evluation in `plms.txt`, and open source the code to generate various mutated models in `mutatePLMs.py`.


