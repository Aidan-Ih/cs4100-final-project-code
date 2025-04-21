# CS4100 Final: Learned Bloom Filters


### **Training Model**:
run `train_forest.py` after downloading the kaggle dataset. In the same directory, it will save a .pkl model of the forest and a csv with columns for each test key, it's label, and the score the model predicted.

### **Running Tests**:
run `test_bloom.py` or `test_learned_bloom.py` to insert all positive data points into the filter, then query all negatives to count the number of false positives
