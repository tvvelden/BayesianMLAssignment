import numpy as np

class BayesianRegression():
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha:float=1., beta:float=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim:int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        bayesian update of parameters given training dataset

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """

        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        """
        return mean (and standard deviation) of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable
        return_std : bool, optional
            flag to return standard deviation (the default is False)
        sample_size : int, optional
            number of samples to draw from the predictive distribution
            (the default is None, no sampling from the distribution)

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        y_sample : (N, sample_size) np.ndarray
            samples from the predictive distribution
        """

        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y



    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

class EmpiricalBayesRegression(BayesianRegression):
    """
    Empirical Bayes Regression model
    a.k.a.
    type 2 maximum likelihood,
    generalized maximum likelihood,
    evidence approximation

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    evidence function p(t|X,alpha,beta) = S p(t|w;X,beta)p(w|0;alpha) dw
    """

    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha, beta)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        maximization of evidence function with respect to
        the hyperparameters alpha and beta given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        max_iter : int
            maximum number of iteration
        """
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for it in range(max_iter):
            params = [self.alpha, self.beta]
            if not(it):
                alpha_ = [self.alpha]
                beta_ = [self.beta]
            else:
                alpha_ = np.hstack((alpha_, self.alpha))
                beta_ = np.hstack((beta_, self.beta))
            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t) 

            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float(
                (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)
        self.alpha_ = alpha_
        self.beta_ = beta_

    def _log_prior(self, w):
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

    def _log_posterior(self, X, t, w):
        return self._log_likelihood(X, t, w) + self._log_prior(w)

    def log_evidence(self, X:np.ndarray, t:np.ndarray):
        """
        logarithm or the evidence function

        Parameters
        ----------
        X : (N, D) np.ndarray
            indenpendent variable
        t : (N,) np.ndarray
            dependent variable
        Returns
        -------
        float
            log evidence
        """
        N = len(t)
        D = np.size(X, 1)
        return 0.5 * (
            D * np.log(self.alpha) + N * np.log(self.beta)
            - np.linalg.slogdet(self.w_precision)[1] - D * np.log(2 * np.pi)
        ) + self._log_posterior(X, t, self.w_mean)

class EM(BayesianRegression):
    """
    Expectation Maximization (EM)

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha, beta)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=1000):
        """
        maximization of evidence function with respect to
        the hyperparameters alpha and beta given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        max_iter : int
            maximum number of iteration
        """
        K=np.size(X, 1)
        eye = np.eye(K)
        N = len(t)
        for it in range(max_iter):
            params = [self.alpha, self.beta]
            if not(it):
                alpha_ = [self.alpha]
                beta_ = [self.beta]
            else:
                alpha_ = np.hstack((alpha_, self.alpha))
                beta_ = np.hstack((beta_, self.beta))
            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)            
            self.alpha = float(K / (np.trace(np.linalg.inv(w_precision))
                                    +np.sum(w_mean ** 2)))
            self.beta = float(
                N / (np.sum(np.square(t - X @ w_mean))
                     +np.trace(X @ np.linalg.inv(w_precision) @ X.T))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)
        self.alpha_ = alpha_
        self.beta_ = beta_

    def _log_prior(self, w):
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

    def _log_posterior(self, X, t, w):
        return self._log_likelihood(X, t, w) + self._log_prior(w)

    def log_evidence(self, X:np.ndarray, t:np.ndarray):
        """
        logarithm or the evidence function

        Parameters
        ----------
        X : (N, D) np.ndarray
            indenpendent variable
        t : (N,) np.ndarray
            dependent variable
        Returns
        -------
        float
            log evidence
        """
        N = len(t)
        D = np.size(X, 1)
        return 0.5 * (
            D * np.log(self.alpha) + N * np.log(self.beta)
            - np.linalg.slogdet(self.w_precision)[1] - D * np.log(2 * np.pi)
        ) + self._log_posterior(X, t, self.w_mean)

class VB(BayesianRegression):
    """
    Variational Bayes (VB)

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha, beta)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=10000):
        """
        maximization of evidence function with respect to
        the hyperparameters alpha and beta given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        max_iter : int
            maximum number of iteration
        """
        a,b,c,d = 1e-6,1e-6,1e-6,1e-6
        K=np.size(X, 1)
        N = len(t)
        self.alpha = self.alpha * np.ones(K)
        for it in range(max_iter):
            params = [np.mean(self.alpha), self.beta]
            if not(it):
                alpha_ = [self.alpha]
                beta_ = [self.beta]
            else:
                alpha_ = np.vstack((alpha_, self.alpha))
                beta_ = np.hstack((beta_, self.beta))
            w_precision = np.diag(self.alpha) + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)   
            
            at = a + 0.5
            bt = b + 0.5 * (np.diag(np.linalg.inv(w_precision)) + w_mean ** 2)
            self.alpha = at/bt
            
            ct = c + N/2
            dt = d + 0.5 * (np.sum(np.square(t - X @ w_mean))
                     +np.trace(X @ np.linalg.inv(w_precision) @ X.T))
            self.beta = ct/dt
            
            if np.allclose(params, [np.mean(self.alpha),self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)
        self.alpha_ = alpha_
        self.beta_ = beta_ 
                
    def _log_prior(self, w):
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

    def _log_posterior(self, X, t, w):
        return self._log_likelihood(X, t, w) + self._log_prior(w)

    def log_evidence(self, X:np.ndarray, t:np.ndarray):
        """
        logarithm or the evidence function

        Parameters
        ----------
        X : (N, D) np.ndarray
            indenpendent variable
        t : (N,) np.ndarray
            dependent variable
        Returns
        -------
        float
            log evidence
        """
        N = len(t)
        D = np.size(X, 1)
        return 0.5 * (
            D * np.log(self.alpha) + N * np.log(self.beta)
            - np.linalg.slogdet(self.w_precision)[1] - D * np.log(2 * np.pi)
        ) + self._log_posterior(X, t, self.w_mean)