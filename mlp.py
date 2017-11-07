# mlp.py
# -------------

# mlp implementation
import util
import numpy
import math
from random import random
PRINT = True

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
    self.network = []

  def makeNeuralNet(self, trainingData, trainingLabels):
    network = list()
    numIn = len(trainingData[0])      #index of training data sets the number of input nodes
    numOut = len(self.legalLabels)    #possible discrete results, labels
    numHidden = 32                    #personal preference for middle layer

    hidden_layer = [{'weights':[random() for i in range(numIn + 1)]} for i in range(numHidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(numHidden + 1)]} for i in range(numOut)]
    network.append(output_layer)

    return network
      
  def train(self, trainingData, trainingLabels, validationData, validationLabels ):
    network = self.makeNeuralNet(trainingData, trainingLabels)
    self.network = network
    guesses = []

    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
        "*** YOUR CODE HERE ***"
        rate = .5                                       #set learning weight for use during weight assignment

        trainingList = list()  
        trainingList.extend(trainingData[i].values())    #add data points to local list
        temp = trainingLabels[i]
        trainingList.append(temp)                       #as well as their corresponding values
        
        #forward propogation

        inputs = trainingList
        for layer in network:                         
          temp = []
          for neuron in layer:
            weights = neuron['weights']
            activation = weights[-1]
            for i in range(len(weights)-1):
              activation += weights[i] * inputs[i]     #activation of neuron during forward propogation
            neuron['output'] = numpy.tanh(activation/100.0)
            #neuron['output'] = 1.0/(1.0 + exp(-activation))
            temp.append(neuron['output'])
          inputs = temp                                #update the output weights by the activation value in order



        eValue = [0 for i in range(len(self.legalLabels))]
        eValue[trainingList[-1]] = 1
        error = 0
        for i in range(len(eValue)):
          error += sum([(eValue[i] - inputs[i]) ** 2 for i in range(len(eValue))])
        
        #backwards propogation

        for i in reversed(range(len(network))):
          layer = network[i]
          errorList = list()
          for j in range(len(layer)):
            if(i == len(network) - 1):                          #for last item in reversed network
              for j in range(len(layer)):
                neuron = layer[j]
                errorList.append(eValue[j] - neuron['output'])  #adjust the expected Value by the of the outputs
            else:
              error = 0.0
              for neuron in network[i+1]:
                weights = neuron['weights']
                error += weights[j] * neuron['delta']
              errorList.append(error)
         
          for j in range(len(layer)):
            neuron = layer[j]
            output = neuron['output']
            neuron['delta'] = errorList[j] * (1 - output**2)

        #update weights

        for i in range(len(network)):
          inputs = trainingList[:-1]
          if i != 0:                        #catch no input condition and use outputs as inputs
            inputs = [neuron['output'] for neuron in network[i-1]]

          for neuron in network[i]:
            for j in range(len(inputs)):
              neuron['weights'][j] += rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += rate * neuron['delta']

      #util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"

      classificationList = list()
      classificationList.extend(datum.values())
      guess = random()
      classificationList.append(guess) 

      network = self.network

      #forward propogation

      inputs = classificationList
      for layer in network:                         
        temp = []
        for neuron in layer:
          weights = neuron['weights']
          activation = weights[-1]
          for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]             #activation of neuron during forward propogation
          neuron['output'] = numpy.tanh(activation/100.0)    #transfer of neuron value
          #neuron['output'] = 1.0/(1.0 + exp(-activation))
          temp.append(neuron['output'])

        inputs = temp                                        #update the output weights by the activation value in order
      prediction = inputs.index(max(inputs))
      guesses.append(prediction)
      #util.raiseNotDefined()
    return guesses
