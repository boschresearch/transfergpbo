# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import numpy as np
from typing import Tuple, Dict, Hashable
from sklearn.preprocessing import StandardScaler

from GPy.models import GPRegression
from GPy.kern import RBF, Kern

from transfergpbo.models import InputData, TaskData, Model
from transfergpbo.models.utils import is_pd, nearest_pd


class GPBO(Model):
    """Gaussian-process (GP) regression object.

    Wrapper around `GPy.models.GPRegression` that (i) performs data normalization before
    training the Gaussian process and de-normalization when predicting, and (ii) allows
    for prediction of an empty GP.
    """

    def __init__(
        self,
        kernel: Kern = None,
        noise_variance: float = 1.0,
        normalize: bool = True,
        **options: dict
    ):
        """Initialize the method.

        Args:
            kernel: The type of kernel of the GP. Defaults to squared exponential
                without automatic relevance determination.
            noise_variance: The variance of the observation noise.
            normalize: Train the model on normalized (`=True`) or original (`=False`)
                data.
            **options: Training arguments for `GPy.models.GPRegression`.
        """
        super().__init__()
        self._kernel = kernel if kernel is not None else RBF(1)

        self._noise_variance = np.array(noise_variance)
        self._gpy_model = None

        self._normalize = normalize
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None

        self._options = options

    @property
    def kernel(self):
        """Return GPy kernel in the normalized space."""
        return self._kernel

    @property
    def x_normalizer(self):
        """Return the x-normalizer of the data."""
        return self._x_normalizer

    @property
    def y_normalizer(self):
        """Return the y-normalizer of the data."""
        return self._y_normalizer

    @property
    def noise_variance(self):
        """Return noise variance."""
        return self._noise_variance

    @kernel.setter
    def kernel(self, kernel: Kern):
        """Assign a new kernel to the GP.

        Args:
            kernel: the new kernel to be assigned.

        """
        self._kernel = kernel.copy()
        if self._gpy_model:
            # remove the old kernel from being a parameter of `gpy_model`
            self._gpy_model.unlink_parameter(self._gpy_model.kern)
            del self._gpy_model.kern
            self._gpy_model.kern = kernel  # assign new kernel
            # add the new kernel to the param class
            self._gpy_model.link_parameter(kernel)
            # re-cache the relevant quantities of the model
            self._gpy_model.parameters_changed()

    def meta_fit(self, metadata: Dict[Hashable, TaskData], **kwargs):
        pass

    def fit(
        self,
        data: TaskData,
        optimize: bool = False,
    ):
        self._X = np.copy(data.X)
        self._y = np.copy(data.Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)

        if self._normalize:
            _X = self._x_normalizer.fit_transform(_X)
            _y = self._y_normalizer.fit_transform(_y)

        if self._gpy_model is None:
            self._gpy_model = GPRegression(
                _X, _y, self._kernel, noise_var=self._noise_variance
            )
        else:
            self._gpy_model.set_XY(_X, _y)

        if optimize:
            optimize_restarts_options = self._options.get(
                "optimize_restarts_options", {}
            )

            kwargs = copy.deepcopy(optimize_restarts_options)

            if "verbose" not in optimize_restarts_options:
                kwargs["verbose"] = False

            self._gpy_model.optimize_restarts(**kwargs)

        self._kernel = self._gpy_model.kern.copy()
        self._noise_variance = self._gpy_model.likelihood.variance.values

    def _denormalize(
        self, mean: np.ndarray, var: np.ndarray, return_full: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize the mean and variance prediction.

        Args:
            mean: Mean of the GP prediction. `shape = (n_points, 1)`.
            var: Variance or covariance of the GP prediction. `shape = (n_points, 1)` or
                `(n_points, n_points)`
            return_full: Returns diagonal of covariance matrix if set to `False`.

        Returns:
            - mean - Normalized mean. `shape = (n_points, 1)`
            - var - Normalized variance or covariance. `shape = (n_points, 1)` or
                `(n_points, n_points)`
        """
        mean, var = copy.deepcopy(mean), copy.deepcopy(var)

        mean = self._y_normalizer.inverse_transform(mean)

        if return_full & (mean.shape[1] > 1):
            var = var[..., np.newaxis] * self._y_normalizer.var_
        else:
            var *= self._y_normalizer.var_

        return mean, var

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, var = self._raw_predict(data, return_full, with_noise)

        if self._X is None:
            return mean, var

        if self._normalize:
            mean, var = self._denormalize(mean, var, return_full=return_full)

        return mean, var

    def _raw_predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict functions distribution(s) for given test point(s) without taking into
        account data normalization. If `self._normalize` is `False`, return the same as
        `self.predict()`.

        Same input/output as `self.predict()`.
        """
        _X_test = data.X.copy()

        if self.X is None:
            mu = np.zeros((_X_test.shape[0], 1))
            cov = self._kernel.K(_X_test)
            var = np.diag(cov)[:, None]
            return mu, cov if return_full else var

        if self._normalize:
            _X_test = self._x_normalizer.transform(_X_test)

        # ensure that no negative variance is predicted
        mu, cov = self._gpy_model.predict(
            _X_test, full_cov=return_full, include_likelihood=with_noise
        )
        if return_full:
            if not is_pd(cov):
                cov = nearest_pd(cov)
        else:
            cov = np.clip(cov, 1e-20, None)
        return mu, cov

    def predict_posterior_mean(self, data: InputData) -> np.ndarray:
        r"""Perform model inference.

        Predict the posterior mean of the latent distribution `f` for given test points.
        Achieves the same as `self.predict(data)[0]` but is much faster.
        Scales as $\mathcal{O}(n)$, where $n$ is the number of training points. Useful
        when the (co-)variance prediction is not needed. Computing the latter scales as
        $\mathcal{O}(n^2)$.

        Args:
            data: Input data to predict on. `shape = (n_points, n_features)`

        Returns:
            The mean prediction. `shape = (n_points, 1)`
        """
        _x = data.X.copy()
        if self._X is None:
            return np.zeros(_x.shape)
        _X = self._X.copy()
        if self._normalize:
            _x = self._x_normalizer.transform(_x)
            _X = self._x_normalizer.transform(_X)
        mu = self._kernel.K(_x, _X) @ self._gpy_model.posterior.woodbury_vector
        if self._normalize:
            mu = self._y_normalizer.inverse_transform(mu)
        return mu

    def predict_posterior_covariance(self, x1: InputData, x2: InputData) -> np.ndarray:
        """Perform model inference.

        Predict the posterior covariance between `(x1, x2)` of the latent distribution
        `f`. In case `x1 == x2`, achieves the same as
        `self.predict(x1, return_full=True)[1]`.

        Args:
            x1: Input data to predict on. `shape = (n_points_1, n_features)`
            x2: Input data to predict on. `shape = (n_points_2, n_features)`

        Returns:
            Predicted covariance for every input. `shape = (n_points_1, n_points_2)`
        """
        _X1 = x1.X.copy()
        _X2 = x2.X.copy()

        if self._X is None:
            cov = self._kernel.K(_X1, _X2)
            return cov

        if self._normalize:
            _X1 = self._x_normalizer.transform(_X1)
            _X2 = self._x_normalizer.transform(_X2)

        cov = self._gpy_model.posterior_covariance_between_points(
            _X1, _X2, include_likelihood=False
        )

        if self._normalize:
            cov *= self._y_normalizer.var_

        return cov

    def compute_kernel(self, x1: InputData, x2: InputData) -> np.ndarray:
        """Evaluate the kernel matrix for desired input points.

        Wrapper around `self.kernel.K()` that takes care of normalization and allows
        for prediction of empty GP.

        Args:
            x1: First input to be queried. `shape = (n_points_1, n_features)`
            x2: Second input to be queried. `shape = (n_points_2, n_features)`

        Returns:
            Kernel values at `(x1, x2)`. `shape = (n_points_1, n_points_2)`
        """
        _x1, _x2 = np.copy(x1.X), np.copy(x2.X)
        if self._normalize and self._X is not None:
            _x1 = self._x_normalizer.transform(_x1)
            _x2 = self._x_normalizer.transform(_x2)
        return self._kernel.K(_x1, _x2)

    def compute_kernel_diagonal(self, data: InputData) -> np.ndarray:
        """Evaluate diagonal of kernel matrix for desired input points.

        Much faster than `compute_kernel()` in case only the diagonal is needed.
        Wrapper around `self.kernel.Kdiag()` that takes care of normalization and
        allows for prediction of empty GP.

        Args:
            data: Input to be queried. `shape = (n_points, n_features)`

        Returns:
            Kernel diagonal. `shape = (n_points, 1)`
        """
        _x = np.copy(data.X)
        if self._normalize and self._X is not None:
            _x = self._x_normalizer.transform(_x)
        return self._kernel.Kdiag(_x).reshape(-1, 1)

    def sample(
        self, data: InputData, size: int = 1, with_noise: bool = False
    ) -> np.ndarray:
        """Perform model inference.

        Sample functions from the posterior distribution for the given test points.

        Args:
            data: Input data to predict on. `shape = (n_points, n_features)`
            size: Number of functions to sample.
            with_noise: If `False`, the latent function `f` is considered. If `True`,
                the observed function `y` that includes the noise variance is
                considered.

        Returns:
            Sampled function value for every input. `shape = (n_points, size)`
        """
        mean, cov = self.predict(data, return_full=True, with_noise=with_noise)
        mean = mean.flatten()
        sample = np.random.multivariate_normal(mean, cov, size).T
        return sample
