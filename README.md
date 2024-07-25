# brain-multimodal

### [Project Page](https://neuro-vis-lang.github.io) | [Paper](https://arxiv.org/abs/2406.14481)

[Vighnesh Subramaniam](https://vsubramaniam851.github.io),
[Colin Conwell](https://colinconwell.github.io/),
[Christopher Wang](https://czlwang.github.io/),
[Gabriel Kreiman](https://klab.tch.harvard.edu/),
[Boris Katz](https://people.csail.mit.edu/boris/boris.html),
[Ignacio Cases](https://stanford.edu/~cases/),
[Andrei Barbu](http://0xab.com/)

This is the main code base of the paper "Revealing Vision-Language Integration of the Brain Using Multimodal Networks", accepted in ICML 2024. 

## Installation
* pytorch >= 1.12.1

```
pip install -r requirements.txt
```

* Install the [SLIP Models](https://github.com/facebookresearch/SLIP) by cloning the repo to the `vil_embeds` directory.

## Data
The data is based on the [Brain Treebank](https://braintreebank.dev). In order to streamline this process, we zip the data with all the necessary preprocessing. This is available [here]()(This is coming soon!). 

The directory should be named `data-by-subject`. In this directory, there are several subdirectories named according to a subject ID. Each directory contains a parquet file which contains the averaged neural response from each electrode across 161 time bins. We also include the stimuli in CSVs. The language-aligned CSV contains the words and paths to the corresponding frame. The vision-aligned CSVs contains paths to the scenes and the closest sentence.  

## Regressions

To run a regression, select a subject, model, dataset alignment and layer. You can use the subdirectories to choose a subject. The dataset alignment can either be language-aligned or vision-aligned. More details on these design choices are in the paper.

To find an available model, check out `model_layer_dict.py`. This has a dictionary with all models and the corresponding layers. If you just want the best performing layer (usually the final layer), choose the the layer subset with `best` in the string. If you want to give a list of layers, add the layer names to the python file, run `python model_layer_dict.py` to get the model layer dictionary, and run the regression with this sublist. If you want to use a randomly initialized model, use the `-r` flag. 

First run

```
python model_layer_dict.py
```

Then, run the following:

```
python run_regression -s [SUBJECT] -t trial000 -a [ALIGNMENT] -w 200 -mn [MODEL_NAME] -mo [MODEL_EXTRACTION_OUTPUT] [-r]
```

The results are saved to a parquet file which can be opened in pandas.

## Bootstrapping Analysis

To run a boostrapping analysis, you can run

```
python bootstrap_test.py -s [SUBJECT] -t trial000 -a [ALIGNMENT] -w 200 -mn [MODEL_NAME] -mo [MODEL_EXTRACTION_OUTPUT] [-r]
```

If you want to run the analysis correctly, you should save your bootstrap indices so that the same resampled inputs are used across subjects and models. The `bootstrap_test.py` file does use using the `resample_indices_caller` function. Uncomment that line and re-run the analysis on your own data.

## Comparison Analysis

We include the a python notebook, `comparison_analysis.ipynb`, to walk through our methodology to compare network scores against each other per electrode. We will also include a raw results from our analysis to give the final output structure needed to perform the analysis if you want to do so on your own data.
