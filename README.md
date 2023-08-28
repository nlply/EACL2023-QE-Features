# EACL2023-QE-Features

>Code for the paper *Mining Effective Features Using Quantum Entropy for Humor Recognition* for EACL 2023.



You need to download the **glove.6B.50d.txt** file from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it under the project.


Running `train.py` will give you the [dataset_features.csv](dataset_features.csv) file, run the script as follows:
```
python train.py --dataset dataset.csv --word_embedding glove.6B.50d.txt --features dataset_features.csv
```


Running `cross_validation.py` for cross-validation, run the script as follows:
```
python cross_validation.py --dataset dataset.csv --word_embedding glove.6B.50d.txt --features dataset_features.csv
```

If you want boost a content-based classifier, run the script as follows:

```
python cross_validation.py --useglove --dataset dataset.csv --word_embedding glove.6B.50d.txt --features dataset_features.csv
```
