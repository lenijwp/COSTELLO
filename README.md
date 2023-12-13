# Testing4LMaaS
## TL;DR
COSTELLO: Contrastive Testing for Large Language Model as a Service Embeddings

## Repo Structure
The core file is as follows:
1. `data` : It contains the dataset we used and the test suite we generated in our experiment.
2. `Testing.py` : It contains the source codes of the testing stage in COSTELLO.
3. `Evaluation-DCs.py` : It contains the source codes about how to construct the ground truth and evalute the test results by such as precsions.
4. `TrainDCs.py` : It contains the source codes about how to train 14 downstream  classifiers.
5. `Fixing.py` : It contains the source codes of our fixing experiment.


## Setup

COSTELLO requires specific versions of some packages, which may conflict with others. Thus, a virtual environment is recommended.
Firstly, you need to install python 3.8 and PyTorch 1.10.2 from [here](https://pytorch.org/get-started/locally/). Then install other packages, run:
```
pip install -r ./requirements.txt
```
What's more, if you want to run the generator module, please install [this third-party library](https://github.com/GEM-benchmark/NL-Augmenter) manually.


## Usage
Here, we give a demo to show how to use our codes.

To test the LLMs:
```
python Testing.py --contrastset ./data/ctset1.json --plm bert-base-uncased --gpu 1 --output ./data/results/bert.json --norm l2 --thres zero
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
|--norm | the distance metric | ['l2','l1','cos'] |
|--thres | the threshold | ['min','1sigma','2sigma','zero'] |
|--epoch | the training epochs of classifiers | 100 |


We provide the test suite used by our evluation at `./data/ctest1.json`, if you want to generate your own test suite, you can do like following:
```
python buildContrastset --trans ./data/SA.json --input  ./data/sst_train.json --output XXX.json
```

## Reproducing Tips:
We list the LLMs we employed in our evluation in `plm.list`, and LLMs we used in our fixing experiments in `plmsforfix.list`.
We provide pipeline scripts to automatically run our experiments.
Yor can reproduce our testing and evaluation experiment as:
```
python script.py
```
And you can reproduce our fixing experiment as:
```
python script-withfix.py
```


