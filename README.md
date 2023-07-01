Welcome to the code repository of our paper. 

## Workflow

Replicating our experiments is relatively straightforward. At a high-level, it entails performing the following steps:

1) Download the datasets from the source (by following the instructions provided in "data-acquisition.md" document)
   * Solely for CTU13, remember to follow the "prelabeling" steps as well!
2) Apply the preprocessing steps for each of the 5 datasets we considered (for which we provide the code notebooks in the preprocessing folder)
   *  For convenience, we can provide _upon request_ the preprocessed data, as well as the original source data (in the unfortunate event that it is moved or taken down)
3) After ensuring that the datasets match (compute the SHA256 and check it against the value provided in each subfolder of the "data" folder), you can proceed to run the notebooks in the "evaluation" folder.

Have fun!

## Requirements

We developed our code between the end of 2021 and the beginning of 2022. We used the following Python libraries:

* scikit-learn: version 0.23
* pandas: version 1.0.5
* numpy: version 1.18
* mlxtend: 0.19.0
* ipaddress: 1.0

All running on Python 3.8. The code provided in this repository may slightly differ due to some necessary adaptations we applied to test our code on different systems. 


Specifically, the code provided in this repository has been tested to be working on a venv which can be setup as follows (using anaconda):

```
conda create -n pragAss python=3.10.11 anaconda
conda activate pragAss
conda install -n pragAss mlxtend=0.22.0 sklearn=1.2.2 numpy=1.24.3 pandas=1.5.3 ipaddress=1.0
```

(it also works on ```pandas=2.0.2```, which can be installed via ```pip```)