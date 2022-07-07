# TSEvo: Counterfactuals for Time Series Classification 
# Installation
To use TSEvo isntall:
```
pip install .
```
For rerunning the experiments, installing the requirements.txt is necessary.

# Usage
Entry point to TSEvo is ```CounterfactualExplanation.py```. 
```python
from TSEvo.CounterfactualExplanation import Explanation

model = # Your ML Model returning a probability distribution
data = #Additionally available in structure (x,y) 
backend= #'Torch' or 'Tensorflow'

exp = Explanation(model,data,backend)

original_x= # Data to explain
y= # Prediction to explain
target_y= # can be None
transformer= # choice of ['authentic_opposing_information','mutate_both','mutate_mean','frequency_band_mapping']

cf=ep.explain_instance(original_x,y, target_y= None,transformer = 'authentic_opposing_information')

```

For more examples and available plots, refer to the jupyter notebooks.
- For Multivariate Data `Multidimensional_Evo.ipynb`
- For Univariate Data `Univariate_Evo.ipynb`
- Tensorflow: `GunPoint_tensorflow.ipynb`
- Benchmarking: `Benchmarking.ipynb`


# Rerun Evaluation
Download UCR Dataset from : https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/ into ./data and unzip.
## Rerun Training Classification Model
UCR: `python Train_ResNet_UCR.py`
UEA: `python Train_ResNet_UEA.py`
## Rerun Counterfactual Generation
UCR: `python Run_UCR.py`
UEA: `python Run_UEA.py`
## Rerun Baselines
UCR: `python Benchmarking_UCR.py`
UEA: `python Benchmarking_UEA.py`
## Refactor Tables in Visualizations from Paper
Compare Mutation: `python PaperEval.py`
Baselines: `python Paper_Vis_Baslines.py`
