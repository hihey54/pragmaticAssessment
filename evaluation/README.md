This folder contains five Jupyter Notebooks, each referring to a specific dataset (out of 5) that we considered in our paper. 

# High Level

The code in each of these Notebooks is very similar: the only changes are the first ~7 cells, which focus on loading the data, perform some preliminary operations (including assigning the parameter values), and split the data into train:test.

Starting from the ~8th cell, the code is the same for all Notebooks. 

We recall that, for our Pragmatic Assessment, we repeated our experiments 100s of times, and we averaged all results at the end. This can be done by extending these notebooks with a simple ```for loop```, which repeats all operations (starting from the 8th cell) a given number of times. Of course, this also requires to store the results of each run in, e.g., a ```list```. 

The case in which your results will most likely differ from ours are those entailing the "limited" data availability scenario: the sheer randomness of the chosen samples can lead to significantly different performance. This is why we repeated these evaluations 1000 times.


# Low Level

We provide more details on our Notebooks

## Preliminary operations (cells 1--7)

We explain what each of the first 7 cells does.

1. We first import some libraries and some functions that we will use for our experiments. We also specify the ```root_folder```, which contains the data used for the Notebook (remember that it must have been already preprocessed!)
2. We specify various parameters. A description of each parameter is provided in the cell itself.
3. We read the input data, including the benign and malicious files. At the end of this cell, there will M+1 dataframes: ```benign_df```; and M dataframes, each corresponding to a specific class of malicious samples.
4. We create the train and test partitions for each of the M+1 dataframes.
5. We merge all dataframes together and factorize some categorical variables (if present); then, we re-create the "attack specific" dataframes; next, we split each of these into a train and test partition.
6. We specify the features included in the "complete" (i.e., the ```features``` list) and "essential" (i.e., the ```small_features``` list) feature set
7. We create the dataset containing 
the adversarial samples. We do this 
by taking the dataset containing the _test_ samples of each attack, and change them (depending on the provided parameters). 

## Real evaluation 

The operations henceforth are straightforward: we train and test each "ML pipeline" described in our paper. Specifically:

* A stand-alone binary classifier
* The multi-class classifier (in a cascade with the previous "binary" classifier
* A stand-alone multi-class classifier
* The various ensembles
  * We first train M weak learners (one per attack)
  * Then we test them by combining their predictions with either a logical or, or majority voting
  * Finally, we consider a stacked clasifier

We first do the above for the "Complete" feature set. Then, we do it in the "open world" setting, which simulates what happens if any given ML pipeline is trained on M-1 attacks, and tested on the leftout attack; of course, we repeat this M times and average the results.
Finally, we repeat the first set of operations a second time but by considering the "Essential" feature set, and we also assess the adversarial robustness of each ML pipeline. 

## Results

At the end of the Notebook, we print out all results. The code is self explanatory.

We always report the confusion matrices, the tpr, the fpr, the inference and training runtime, and also the accuracy (the latter only for the multiclass classifiers).

