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
from typing import Dict, Hashable, Tuple

import GPy.likelihoods
import numpy as np
from GPy.core import GP
from GPy.kern import Kern
from transfergpbo.models import InputData, TaskData, GPBO
from transfergpbo.models.utils import (
    is_pd,
    nearest_pd,
    CrossTaskKernel,
)


class HGP(GPBO):
    r"""Hierarchical GP, a GP-based transfer-learning algorithm.

    Hierarchical GP is a kernel method that models the source and target data
    hierarchically in exactly the same way as SHGP. The only
    difference to SHGP is hyperparameter optimization. In particular, Hierarchical GP
    optimizes all GP hyperparameters jointly. By contrast, SHGP optimizes the
    hyperparameters sequentially in the hierarchy. Hierarchical GP models the data with
    a kernel of the form

    $$
        k((x, \nu), (x', \nu')) = k_s + \delta_{\nu=t}\delta_{\nu'=t}k_t,
    $$

    where $\nu$ is the task label, $\nu=\{s, t\}$, and $\delta$ equals one if the
    argument holds and zero otherwise.

    Extension to multiple source tasks is done by keeping the hierarchical
    structure of the kernel

    $$
        k((x, \nu), (x', \nu')) = \sum_{i=s_1,s_2,\ldots,t} \delta_{\nu\geq i} \delta_{\nu'\geq i} k_i,
    $$

    where $\nu = \{ s_1, s_2, \ldots, t \}$.

    Hierarchical GP is a powerful but computationally expensive method since (i) it
    scales cubically with the total number of data points and (ii) the number of
    hyperparameters scales linearly with the number of tasks. Hierarchical GP is
    expected to outperform SHGP in the limit of ultra-low number of source points,
    which are not sufficient to provide a descriptive source model.
    """

    def __init__(
        self,
        kernel: Kern = None,
        noise_variance: float = 1.0,
        normalize: bool = True,
        **options: dict,
    ):
        super().__init__(kernel, noise_variance, normalize, **options)
        self._normalize = normalize
        self._meta_data = None
        self._n_sources = None
        self._likelihood = None
        self._default_kernel = self._kernel.copy()

    def meta_fit(
        self,
        source_datasets: Dict[Hashable, TaskData],
        **kwargs,
    ):
        self._n_sources = len(source_datasets)
        n_features = list(source_datasets.values())[0].X.shape[-1]
        data = copy.deepcopy(source_datasets)

        x_all, y_all, k_all = [], [], []
        for i, (source_id, source_data) in enumerate(data.items()):
            x_all.append(
                np.concatenate(
                    [source_data.X, np.full((source_data.X.shape[0], 1), i)], axis=-1
                )
            )
            y_all.append(source_data.Y)

        x_all = np.concatenate([*x_all], axis=0)
        y_all = np.concatenate([*y_all], axis=0)
        self._meta_data = (x_all, y_all)

        # Kernel needs to operate on the full feature vector
        np.testing.assert_equal(self._default_kernel.active_dims, np.arange(n_features))

        self._kernel = self._get_kernel(
            num_features=n_features, num_source_tasks=self._n_sources
        )

        # Create mixture of different likelihoods
        likelihoods = [
            GPy.likelihoods.Gaussian(variance=self._noise_variance)
            for _ in range(self._n_sources + 1)
        ]
        self._likelihood = GPy.likelihoods.MixedNoise(likelihoods_list=likelihoods)

    def _get_kernel(self, num_features: int, num_source_tasks: int) -> Kern:
        """Construct the kernel for the desired number of features and number of
        source tasks."""
        source_kernels = []
        target_index = num_source_tasks
        task_index_dim = num_features

        for i in range(num_source_tasks):
            task_kernel = self._default_kernel.copy()
            # Hierarchical model: Each layer is correlated with all future layers
            if i >= 1:
                correlated_dims = tuple(range(i, target_index + 1))
                k_delta = CrossTaskKernel(
                    task_indices=correlated_dims, index_dim=task_index_dim
                )
                k_delta.variance.fix()
                task_kernel *= k_delta

            source_kernels.append(task_kernel)

        target_domain = CrossTaskKernel(
            task_indices=target_index, index_dim=task_index_dim
        )
        target_domain.variance.fix()
        target_kernel = self._default_kernel.copy() * target_domain

        return sum(source_kernels, target_kernel)

    def fit(
        self,
        data: TaskData,
        optimize: bool = False,
    ):
        x_t = np.concatenate(
            [data.X, np.full((data.X.shape[0], 1), self._n_sources)], axis=-1
        )

        x_all = np.concatenate([self._meta_data[0], x_t], axis=0)
        y_all = np.concatenate([self._meta_data[1], data.Y], axis=0)

        self._X = np.copy(data.X)
        self._y = np.copy(data.Y)

        _X = x_all
        _y = y_all

        if self._normalize:
            _X[:, :-1] = self._x_normalizer.fit_transform(_X[:, :-1])
            _y = self._y_normalizer.fit_transform(_y)

        # Get output-indexes from data
        Y_metadata = dict(output_index=_X[:, [-1]].astype(int))

        if self._gpy_model is None:
            self._gpy_model = GP(
                X=_X,
                Y=_y,
                kernel=self.kernel,
                likelihood=self._likelihood,
                Y_metadata=Y_metadata,
            )
        else:
            self._gpy_model.Y_metadata = Y_metadata
            self._gpy_model.set_XY(X=_X, Y=_y)

        if optimize:
            optimize_restarts_options = self._options.get(
                "optimize_restarts_options", {}
            )

            kwargs = copy.deepcopy(optimize_restarts_options)

            if "verbose" not in optimize_restarts_options:
                kwargs["verbose"] = False

            self._gpy_model.optimize_restarts(**kwargs)

        self._kernel = self._gpy_model.kern.copy()

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(
            [data.X, np.full((data.X.shape[0], 1), self._n_sources)], axis=-1
        )
        return super().predict(
            InputData(x), return_full=return_full, with_noise=with_noise
        )

    def _raw_predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict functions distribution(s) for given test point(s) without taking into
        account data normalization. If `self._normalize` is `False`, return the same as
        `self.predict()`.

        Same input/output as `self.predict()`.
        """
        _X = data.X.copy()
        _X_test = np.copy(_X)

        if self._X is None:
            mu = np.zeros((_X_test.shape[0], 1))
            cov = self._kernel.K(_X_test)
            var = np.diag(cov)[:, None]
            return mu, cov if return_full else var

        if self._normalize:
            _X_test[:, :-1] = self._x_normalizer.transform(_X[:, :-1])

        # ensure that no negative variance is predicted
        mu, cov = self._gpy_model.predict(
            _X_test,
            full_cov=return_full,
            include_likelihood=with_noise,
            Y_metadata=dict(output_index=np.full(len(_X_test), self._n_sources)),
        )
        if return_full:
            if not is_pd(cov):
                cov = nearest_pd(cov)
        else:
            cov = np.clip(cov, 1e-20, None)
        return mu, cov

    def compute_kernel(self, x1: InputData, x2: InputData) -> np.ndarray:
        _x1 = np.concatenate(
            [x1.X, np.full((x1.X.shape[0], 1), self._n_sources)], axis=-1
        )
        _x2 = np.concatenate(
            [x2.X, np.full((x2.X.shape[0], 1), self._n_sources)], axis=-1
        )
        return super().compute_kernel(InputData(_x1), InputData(_x2))

    def compute_kernel_diagonal(self, data: InputData) -> np.ndarray:
        _x = np.concatenate(
            [data.X, np.full((data.X.shape[0], 1), self._n_sources)], axis=-1
        )
        return super().compute_kernel_diagonal(InputData(_x))
