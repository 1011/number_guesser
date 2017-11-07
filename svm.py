# svm.py
# -------------

# svm implementation
import util
import numpy as np
from sklearn.svm import LinearSVC
import math
PRINT = True

class SVMClassifier:
  """
  svm classifier
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"
    self.clf = LinearSVC(multi_class='ovr')
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):

    trainingList = []

    for i in range(len(trainingData)):
        "*** YOUR CODE HERE ***"
        trainingList.append(trainingData[i].values())
        
    x = np.array(trainingList)
    y = np.array(trainingLabels)
    
    self.clf.fit(x, y)

    
  def classify(self, data ):
    guesses = []
    i = 0
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"
      i = i + 1
      tempArr = np.array([datum.values()])
      guess = self.clf.predict(tempArr)
      guesses.append(guess)

    return guesses
    


