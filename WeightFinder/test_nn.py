from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model
import pandas as pd
import numpy as np
from WeightFinder import nn


class neural(MultipleRegressionAgent):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self,independent_variables, dependent_variable):
        # Initialize your model parameters here
        super().__init__(independent_variables, dependent_variable)

        self.batch_size = 1
        self.hidden_layer_size = 150
        self.input_vector_size = 2  # The length of the input vector
        self.learning_rate = 0.1  # How quickly this nn accepts new weights
        self.W1 = nn.Parameter(self.input_vector_size, self.hidden_layer_size)  # i x h matrix
        self.b1 = nn.Parameter(1, self.hidden_layer_size)  # bias 1
        self.W2 = nn.Parameter(self.hidden_layer_size, self.batch_size)  # h x i matrix (to pass mult. with relu)
        self.b2 = nn.Parameter(1, self.batch_size)  # bias 2

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # f(x)=relu(x⋅W1+b1)⋅W2+b2
        linear1 = nn.Linear(x, self.W1)  # x*W1
        # print("linear1: ",linear1)
        bias1 = nn.AddBias(linear1, self.b1)  # x*W1+b1
        # print("bias1: ", bias1)
        relu = nn.ReLU(bias1)  # relu(x*W1+b1)
        # print("relu: ",relu)
        linear2 = nn.Linear(relu, self.W2)  # relu(x*W1+b1)*W1
        # 10 x 1 and 1 x 784
        # print("linear2: ",linear2)
        bias2 = nn.AddBias(linear2, self.b2)  # relu(x*W1+b1)*W1+b2
        return bias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def run_regression(self, X, y):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        i = 0

        X = X.to_numpy()
        X = nn.Constant(X)

        y = y.to_numpy()
        y = nn.Constant(y)

        while True:


            params = [self.W1, self.b1, self.W2, self.b2]
            loss = self.get_loss(X, y)
            gradients = nn.gradients(loss, params)
            for idx, param in enumerate(params):
                param.update(gradients[idx], -1.0 * self.learning_rate)

            if i % 10 == 0:
                print(params)
            if i >= 100:
                break
            i+=1

"""
        while True:

            for x_node, y_node in zip(X,y):

                params = [self.W1, self.b1, self.W2, self.b2]
                loss = self.get_loss(x_node, y_node)
                gradients = nn.gradients(loss, params)
                for idx, param in enumerate(params):
                    param.update(gradients[idx], -1.0 * self.learning_rate)

                if i % 10 == 0:
                    print(params)
                if i >= 100:
                    break
                i+=1
"""

