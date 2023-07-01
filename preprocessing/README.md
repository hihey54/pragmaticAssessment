This folder contains the code we developed to "preprocess" the source dataset into an appropriate (and "standardized") format for our experiments.

## How to use

There are 5 notebooks in this folder---one per dataset. To run each of the notebooks in this folder, you must have first obtained the corresponding dataset by downloading it from source (and, if necessary, extract it from the archive). At the beginning of each notebook (typically the 3rd cell) you must provide the **folder containing the source files** of the corresponding dataset; you can use the ```list``` "file_names" as a guide to determine which files the notebook is expected to process.

After running the notebook, each source dataset will be transformed into _1+N_ sets, where _N_ is the number of malicious classes in the corresponding dataset (for some datasets, we aggregate samples of "underrepresented" classes into a single class, named "other"). These sets will be saved in the corresponding dataset subfolder (within the "data" folder of this repository). Specifically, the set of all benign samples will be saved in the "flows" folder of the corresponding dataset; and the N sets of malicious samples are saved in the "flows/malicious" folder of the corresponding dataset.

It is recommended to compute the SHA256 of such "flows" folder and compare it with the one provided in this repository: if the SHA256 match, then you can expect that your experiments will yield similar results as the ones we carried out for our paper.

**IMPORTANT**: depending on the version of the libraries used to preprocess the data, the SHA256 can change. This is because some libraries truncate some values (e.g., "2.6000000000000002e-05" can become "2.6e-05" for some versions of pandas). The results are not impacted by such a change.

Upon request (contact giovanni.apruzzese@uni.li) we can provide access to the preprocessed version of all our considered datasets.

## What do these notebooks do?

At a high-level, these notebooks perform the following operations:

* all the samples of a given source dataset (i.e., all the files) are read and loaded into a dataframe
* the labels of some classes are renamed and/or standardized
* the feature "port_type" (denoting a "high/medium/low" port) is computed for every sample (for both src-/dst-port)
* the feature "IP_internal" (denoting whether the IP is from inside or outside the network) is computed (for both src-/dst-IP)
  * We used our own domain knowledge to infer which IP ranges were more likely to be internal or external
* the dataframe is cleaned from infinite or NaN values (which can be troublesome for some classifiers)
* the dataframe is split into 1+N datasets, and saved

Of course, there are some minor differences for each dataset in the way the operations above are implemented.
