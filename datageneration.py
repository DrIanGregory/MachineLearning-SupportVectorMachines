import numpy as np

class Data(object):
    """ Methods surrounding data manipulation."""

    def generate_linearly_separable_data(self,seed=1):
        #-------------------------------------------------------
        # Purpose: Generate (2-dimensional) linearly separable data.
        # -------------------------------------------------------
        np.random.seed(seed)
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data(self,seed=1):
        #-------------------------------------------------------
        # Purpose: Generate (2-dimensional) Non-linearly separable data.
        # -------------------------------------------------------
        np.random.seed(seed)
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data(self,seed=1):
        #-------------------------------------------------------
        # Purpose: Generate (2-dimensional) linearly separable data with some data overlap.
        # -------------------------------------------------------
        np.random.seed(seed)
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_data(X1, y1, X2, y2,percent):
        #-------------------------------------------------------
        # Purpose: Split (2-dimensional) data into training and test. Generate  linearly separable data.
        # Inputs:
        #        X1, y1, X2, y2      : 1*N vector of data.
        #        percent             : scalar indicating how much data should be assigned to test and the remaining will be training.
        # -------------------------------------------------------
        lengthOfData = len(X1)
        cutOff = int(lengthOfData*percent);

        # Training data:
        X1_train = X1[:cutOff]
        y1_train = y1[:cutOff]
        X2_train = X2[:cutOff]
        y2_train = y2[:cutOff]

        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))

        # Test data:
        X1_test = X1[cutOff:]
        y1_test = y1[cutOff:]
        X2_test = X2[cutOff:]
        y2_test = y2[cutOff:]

        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))

        return X_train, y_train, X_test, y_test
