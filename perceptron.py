# perceptron.py
# -------------

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING
    
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"
          trainingRate = 0.02                           # hard coded. Alpha set to 0.02

          keyList = trainingData[i].keys()              # List of Keys for training data
          valueList = trainingData[i].values()          # List of values for training data
          count = util.Counter()                        # Counter vector (x) 
          scores = util.Counter()                       # Counter for scores

          for j in range(len(keyList)):     
            count[keyList[j]] = valueList[j]            # Mapping keys to values

          for j in range(len(self.legalLabels)):
            scores[j] += count * self.weights[j]        # Set score key=> values to weights

          heuristicValue = scores.argMax()              # Set maximum value of score as heuristic
          trueValue = trainingLabels[i]                 # Actual value of sigmoid function output

          if trueValue == heuristicValue:               # No error condition
            continue

          # If error exists, train program
          count.divideAll((1/trainingRate))

          # Set heuristic value of weights to approrpriate 
          self.weights[heuristicValue] -= count
          self.weights[trueValue] += count

    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    sortedKeys = self.weights[label].sortedKeys()
    featuresWeights = sortedKeys[:100]

    return featuresWeights

