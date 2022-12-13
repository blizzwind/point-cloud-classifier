# point-cloud-classifier
A simple point cloud classifier.

### structure
    .
    ├── input
        ├── train (las files for training)
        └── test (las files for predicting)

### note
currently classifying ground point only <br>
las files should only contain ground classification 2 & other classification 31

### run
training
```
python train.py
```
predicting
```
python test.py
```
