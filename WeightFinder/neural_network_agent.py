from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model


class NeuralNetworkAgent(MultipleRegressionAgent):
    """
    #TODO: implement
    Implements a Neural Network regression agent using a homebrewed regression algorithm.
    """
    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)

    def run_regression(self, X, y):
        """
        #TODO: implement
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :return: The list of coefficients calculated by the regression algorithm
        """

        def net(X):
            return mx.nd.dot(X, w) + b

        def square_loss(yhat, y):
            return nd.mean((yhat - y) ** 2)

        
        def SGD(params, lr):
            for param in params:
            param[:] = param - lr * param.grad
        
        epochs = 10
        learning_rate = .0001
        num_batches = num_examples/batch_size

        for e in range(epochs):
            cumulative_loss = 0
            # inner loop
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(model_ctx)
                label = label.as_in_context(model_ctx).reshape((-1, 1))
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output, label)
                loss.backward()
                SGD(params, learning_rate)
                cumulative_loss += loss.asscalar()
            print(cumulative_loss / num_batches)

    

        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X, y)
        print(linear_regression.predict([[3000,2000]]))

        return linear_regression.coef_



