# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features =  np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    total_rows, total_cols = datum.shape[0], datum.shape[1] # Get the total rows and columns
    #A cell in 2D matrix can be connected to 8 neighbors. So, unlike standard DFS(), where we recursively call for all adjacent vertices, here we can recursive call for 4 neighbors only.
    # We keep track of the visited 0s so that they are not visited again.
    number_white_islands = countWhiteIslands(features, total_rows, total_cols)

    extra_features = np.array([0,0,0])
    if number_white_islands == 1:
        extra_features = np.array([1,0,0])
    elif number_white_islands == 2:
        extra_features =  np.array([0,1,0])
    elif number_white_islands > 2:
        extra_features = np.array([0, 0, 1])
    return np.concatenate((features.flatten(), extra_features), axis = 0)

    # features = basicFeatureExtractor(datum)

    # util.raiseNotDefined()

    # return features
def countWhiteIslands(matrix, rows, cols):
    visited = np.zeros((rows, cols))
    count = 0
    for i in range(rows):
        for j in range(cols):
            if not matrix[i][j] and not visited[i][j]:
                DFS(matrix, i, j, rows,cols, visited)
                count += 1
    return count

def DFS(matrix, row, col, rows, cols, visited):
    #Check only the immediate top, bottom, left and right cells.
    neigh_row = [0, 0, 1, -1]
    neigh_col = [-1, 1, 0, 0]
    visited[row][col] = 1
    for k in range(len(neigh_col)):
        if isSafe(matrix, row+neigh_row[k], col+neigh_col[k], rows, cols, visited):
            DFS(matrix, row+neigh_row[k], col+neigh_col[k], rows, cols, visited)

def isSafe(matrix, row, col, rows, cols, vistied):
    # Check if the neighbouring cell is within bounds, is a 0 and has not been visited previously.
    return (row >= 0) and (row < rows) and \
           (col >= 0) and (col < cols) and \
           (not matrix[row][col]) and (not vistied[row][col])

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
