# Testing KNN

## Methodology

I tested my k-nearest neighbors algorithm using 5 different datasets: iris, wine, mushroom, digits, and **another digits dataset that I compiled myself**. When you run `testknn.py`, it will run through a few iterations of each dataset to test its accuracy. For some datasets, it tests with several different test set sizes and several different values for k to find which give greatest accuracy. It also randomly shuffles the data every run so as to not bias the test. A progress bar is shown to demonstrate the progress of the algorithm, and the running time of the algorithm on a specific dataset is printed after each test. In some instances, Sci-Kit Learn's KNN was used to compare accuracy with mine.

### Error logs

In order to better understand the reasons that each set of predictions is either right or wrong, the testing program creates error logs for each test of the algorithm found in the folder `errorlogs`. This keeps track of every discrepancy between the predictions and the actual values in the form:

```
[predicted value] =/= [actual value]
```

## Datasets

### Iris

Sci-Kit Learn's Iris dataset, for predicting what kind of iris a datapoint is. This dataset worked well.

### Wine

Sci-Kit Learn's Wine dataset, for predicting the type of wine a datapoint is. We did not get good results from this dataset, even with L<sub>13</sub> norm (the dataset has dimensionality of size 13), so I tested it with Sci-Kit Learn's KNN, and it also did not score consistenly more than 75%.

### Mushroom

Dataset from UCI's machine learning database from https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/ in the file `agaricus-lepiota.data`, for predicting if a mushroom is poisonous or not (pretty important!). I shuffled this dataset and only took the first the 3000 datapoints from it because it was running very slowly with 8000 datapoints. The data points with this dataset are not numerical; they are catagorical with letters indicating features of the mushroom (e.g. 'x' for convex cap, and 'f' for flat cap). I translated each of these letters to ascii, and the algorithm still worked quite nicely!

### Sci-Kit Learn's Digits

This worked really well.

### My Own Digits

I compiled a set of 1200 handwritten digits from my friends who filled out the `numSheet.pdf` form. I then wrote `cropdigits.py` to pull out 32x32 sized images of each digit and record the grayscale of every pixel in `digits.data`. I did not get good results from this dataset, neither by my own algorithm or Sci-Kit Learn's. I suspected that this might be due to the curse of dimensionality, so I scanned the forms again using the line recognition function to get a cleaner picture, then I modified the cropping tool to shrink the pictures to 16x16 and tried again. This fared no better. I now suspect that the issue is that the location of the digits in the picture differs between the pictures, and further testing is needed.

## Conclusion

My algorithm does well on accuracy, just as well if not better on average then Sci-Kit Learn's KNN algorithm. However, it severely lacks in efficiency. Sci-Kit Learn can do in less than a tenth of a second what takes my algorithm a dozen seconds.