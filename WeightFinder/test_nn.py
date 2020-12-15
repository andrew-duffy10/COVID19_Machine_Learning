import numpy as np
from WeightFinder.data_accumulation import DataAcc
from WeightFinder.learning_agent import MultipleRegressionAgent
import sys
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


class NeuralNetwork(MultipleRegressionAgent):

    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)
        # seeding for random number generation
        np.random.seed(1)
        self.input_scalers = []
        for i in independent_variables:
            self.input_scalers.append(MinMaxScaler(feature_range=(-1,1)))
        self.output_scaler = MinMaxScaler(feature_range=(-1,1))

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((1, 1)) - 1

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            error = training_outputs - output
            #print(output)
            #print(training_outputs)
            #sys.exit(1)
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(np.float128)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


    def run_regression(self, X, y):

        X = X.to_numpy(dtype=np.float128)
        y = np.array([y], dtype=np.float128).T
        print(X)
        self.output_scaler.fit(y)
        self.input_scalers[0].fit(X)
        X = self.input_scalers[0].transform(X)
        y = self.output_scaler.transform(y)
        print(X)
        self.train(X, y, 15000)
        self.variable_weights = self.synaptic_weights

        return self.variable_weights
        #self.train(X.to_numpy(dtype=np.float128), np.array([y.to_numpy()], dtype=np.float128).T, 15000)

        #self.variable_weights = self.synaptic_weights

    def predict(self,entry:list):

        prediction = 0
        print("entry:",entry)
        print("e2:",np.array([entry[0]]).reshape(-1, 1))
        print("weights:",self.synaptic_weights)
        entry = self.input_scalers[0].transform(np.array([entry[0]]).reshape(-1, 1))
        for value, coeff in zip(entry, self.synaptic_weights):
            prediction += (value * coeff)

        print("prediction:")
        return self.output_scaler.inverse_transform([prediction])


        #print(self.think(np.array(X[0])))
        #return self.variable_weights
