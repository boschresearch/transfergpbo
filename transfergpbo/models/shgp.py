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

from typing import Union, Tuple

import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

from GPy.kern import RBF, Kern

from transfergpbo.models import InputData, TaskData, GPBO, MHGP
from transfergpbo.models.utils import (
    is_pd,
    nearest_pd,
    compute_cholesky,
    FixedKernel,
)


class SHGP(MHGP):
    r"""Sequential Hierarchical GP, a GP-based transfer-learning algorithm.

    SHGP improves over MHGP by propagating the uncertainty through
    the stack of GPs. SHGP is a Bayesian technique that models
    the source and target data jointly with a common kernel. The design and API are
    the same as those of MHGP.

    The key idea of SHGP is to include, in addition to the mean,
    also the _uncertainty_ of the source posterior into the prior mean of the target GP.
    For one source and one target dataset, we write

    $$
        p\left[f_t(x)\right] = \mathcal{N}\left[m_t^{(0)}(x), K_t^{(0)}(x, x)\right],
    $$

    where $p\left[f_t(x)\right]$ is the prior distribution of the target function, and
    the prior mean, $m_t^{(0)}(x)$, is the GP of the source posterior with posterior
    mean function $m_s(x)$ and covariance $K_s(x, x)$:

    $$
        m_t^{(0)}(x) \sim \mathcal{N}\left[m_s(x), K_s(x, x)\right].
    $$

    Since the sum of two normally distributed random variables is also normally
    distributed, we obtain the following final expression for the target prior:

    $$
        p\left[f_t(x)\right] = \mathcal{N}\left[m_s(x), K_s(x,x) + K_t^{(0)}(x,x)\right].
    $$

    The difference between the target priors of MHGP and SHGP is a
    new covariance term, $K_s(x,x)$, that describes the noise of the source posterior.
    The uncertainty of the prior is therefore fully propagated to the target model.
    The posterior of the target model is again computed using
    [Gaussian algebra](http://www.gaussianprocess.org/gpml/). Extension to multiple
    source datasets works exactly like in the case of MHGP.

    The propagation of uncertainty notably improves the performance of the algorithm.
    The price to pay is the scalability:

    - SHGP scales as $\mathcal{O}(N_tN_s^2)$ for training and
        $\mathcal{O}(N_s^2)$ for inference, where $N_s(N_t)$ is the number of source
        (target) points.

    - MHGP scales as $\mathcal{O}(N_tN_s)$ for training and $\mathcal{O}(N_s)$
        for inference.
    """

    def __init__(self, n_features, within_model_normalize=True):
        super().__init__(n_features, within_model_normalize=within_model_normalize)
        self._cholesky = None
        self._cholesky_source = []

    def _update_meta_data(self, *gps: GPBO):
        super()._update_meta_data(*gps)
        self._cache_source_quantities()

    def _cache_source_quantities(self):
        """Compute and cache the Cholesky decomposition of the source models"""
        if not self._cholesky_source:
            # the Cholesky of the first source model is not stored
            self._cholesky_source.append(None)
        for i, model in enumerate(self.source_gps[1:], start=1):
            try:
                self._cholesky_source[i]
            except IndexError:
                chol = self._compute_cholesky(idx=i)
                self._cholesky_source.append(chol)

    def _compute_cholesky(self, idx: int = None) -> np.ndarray:
        """Compute and cache the Cholesky decomposition for the `idx`-th GP in the
        stack.

        Counting happens from bottom (=first source GP) to top (=target GP) of the
        stack. If `None`, target GP is assumed.
        """
        idx = self._idx_to_int(idx)
        model, _ = self._idx_to_model_and_cholesky(idx)
        k_sum = self._raw_prior_covariance(
            InputData(model.X), InputData(model.X), idx=idx
        )
        np.fill_diagonal(k_sum, k_sum.diagonal() + model.noise_variance)
        return compute_cholesky(k_sum)

    def _meta_fit_single_gp(
        self,
        data: TaskData,
        optimize: bool,
    ) -> GPBO:
        residuals = self._compute_residuals(data)
        kernel = RBF(self.n_features, ARD=True)
        scale = (
            StandardScaler().fit(residuals).scale_
            if self._within_model_normalize
            else 1.0
        )
        idx = len(self.source_gps)
        if self.source_gps and optimize:
            kernel += self._get_posterior_cov_kernel(InputData(data.X), scale, idx)
        new_gp = GPBO(
            kernel, noise_variance=0.1, normalize=self._within_model_normalize
        )
        new_gp.fit(
            TaskData(X=data.X, Y=residuals),
            optimize=optimize,
        )
        if idx > 0 and optimize:
            new_gp.kernel = new_gp.kernel.rbf
        return new_gp

    def fit(
        self,
        data: TaskData,
        optimize: bool = False,
    ):
        super().fit(data, optimize=False)

        if optimize:  # switch to a sum kernel for hyperparameter optimization
            scale = (
                self.target_gp.y_normalizer.scale_
                if self._within_model_normalize
                else 1.0
            )
            sum_kernel = self.target_gp.kernel + self._get_posterior_cov_kernel(
                InputData(data.X), scale
            )
            self.target_gp.kernel = sum_kernel

            # actual hyperparameter optimization
            super().fit(data, optimize=True)

            self.target_gp.kernel = self.target_gp.kernel.rbf

        self._cholesky = self._compute_cholesky()

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu = self.predict_posterior_mean(data)

        if return_full:
            cov = self.predict_posterior_covariance(data, data)
            if not is_pd(cov):
                cov = nearest_pd(cov)
        else:
            cov = self._predict_posterior_variance(data)
            cov = np.clip(cov, 1e-20, None)

        if with_noise:
            noise_var = self.target_gp.noise_variance
            if self._within_model_normalize and self._X is not None:
                noise_var = noise_var * self.target_gp.y_normalizer.scale_ ** 2
            if return_full:
                np.fill_diagonal(cov, cov.diagonal() + noise_var)
            else:
                cov += noise_var

        return mu, cov

    def predict_posterior_mean(self, data: InputData, idx: int = None) -> np.ndarray:
        idx = self._idx_to_int(idx)
        if idx == 0:
            return self.source_gps[0].predict_posterior_mean(data)
        else:
            return self._posterior_mean(data, idx=idx)

    def _posterior_mean(self, data: InputData, idx: int):
        """Compute the posterior mean.

        Same input/output as `predict_posterior_mean()`.
        """
        model, cholesky = self._idx_to_model_and_cholesky(idx)
        mean_xq = self.predict_posterior_mean(data, idx=idx - 1)
        if model.X is None:
            return mean_xq
        k_sum_X_x = self._raw_prior_covariance(InputData(model.X), data, idx=idx)
        v1 = scipy.linalg.solve_triangular(cholesky, k_sum_X_x, lower=True)
        model_y_norm = model.y
        if self._within_model_normalize:
            model_y_norm = model.y_normalizer.transform(model_y_norm)
        v2 = scipy.linalg.solve_triangular(cholesky, model_y_norm, lower=True)
        v = v1.T @ v2
        if self._within_model_normalize:
            v = model.y_normalizer.inverse_transform(v)
        return mean_xq + v

    def predict_posterior_covariance(
        self, x1: InputData, x2: InputData, idx: int = None
    ) -> np.ndarray:
        idx = self._idx_to_int(idx)
        if idx == 0:
            return self.source_gps[0].predict_posterior_covariance(x1, x2)
        else:
            return self._posterior_covariance(x1, x2, idx=idx)

    def _posterior_covariance(self, x1: InputData, x2: InputData, idx: int):
        """Posterior covariance between two inputs.

        Same input/output as `predict_posterior_covariance()`.
        """
        model, cholesky = self._idx_to_model_and_cholesky(idx)
        k_sum_x1_x2 = self._raw_prior_covariance(x1, x2, idx=idx)
        if model.X is None:
            return k_sum_x1_x2
        if np.array_equal(x1.X, x2.X):
            k_sum_X_x = self._raw_prior_covariance(InputData(model.X), x1, idx=idx)
            v = scipy.linalg.solve_triangular(cholesky, k_sum_X_x, lower=True)
            cov = k_sum_x1_x2 - v.T @ v
            if self._within_model_normalize:
                cov *= model.y_normalizer.scale_ ** 2
            return cov
        k_sum_X_x1 = self._raw_prior_covariance(InputData(model.X), x1, idx=idx)
        k_sum_X_x2 = self._raw_prior_covariance(InputData(model.X), x2, idx=idx)
        v_right = scipy.linalg.solve_triangular(cholesky, k_sum_X_x2, lower=True)
        v_left = scipy.linalg.solve_triangular(cholesky, k_sum_X_x1, lower=True)
        cov = k_sum_x1_x2 - v_left.T @ v_right
        if self._within_model_normalize:
            cov *= model.y_normalizer.scale_ ** 2
        return cov

    def _raw_prior_covariance(
        self, x1: InputData, x2: InputData, idx: int = None
    ) -> np.ndarray:
        """Calculate the prior covariance (=sum kernel) between two inputs in the
        normalized space of `idx`-th GP.

        Args:
            x1: First input to be queried. `shape = (n_points_1, n_features)`
            x2: Second input to be queried. `shape = (n_points_2, n_features)`
            idx: GP index in the stack. Counting happens from bottom (=first source GP)
                to top (=target GP) of the stack.
                Defaults to `None`, which corresponds to target GP.

        Returns:
            Prior covariance at (x1, x2). `shape = (n_points_1, n_points_2)`
        """
        idx = self._idx_to_int(idx)
        model_idx, _ = self._idx_to_model_and_cholesky(idx)
        kernel_idx = model_idx.compute_kernel(x1, x2)  # in normalized space
        posterior_idx_minus_one = self.predict_posterior_covariance(x1, x2, idx=idx - 1)
        if self._within_model_normalize and model_idx.X is not None:
            posterior_idx_minus_one /= model_idx.y_normalizer.scale_ ** 2
        prior_idx = kernel_idx + posterior_idx_minus_one
        return prior_idx

    def _predict_posterior_variance(
        self, data: InputData, idx: int = None
    ) -> np.ndarray:
        """Calculate posterior variance for given input.

        Much faster than `predict_posterior_covariance().diagonal()`.

        Args:
            data: Input to be queried. `shape = (n_points_1, n_features)`
            idx: GP index in the stack. Counting happens from bottom (=first source GP)
                to top (=target GP) of the stack.
                Defaults to `None`, which corresponds to target GP.

        Returns:
            Posterior variance. `shape = (n_points, 1)`
        """
        idx = self._idx_to_int(idx)
        if idx == 0:
            return self.source_gps[0].predict(data)[1]
        else:
            return self._posterior_variance(data, idx=idx)

    def _posterior_variance(self, data: InputData, idx: int = None) -> np.ndarray:
        """Calculate the posterior variance.

        Same input/output as `predict_posterior_variance()`
        """
        model, cholesky = self._idx_to_model_and_cholesky(idx)

        k_sum_x = self._raw_prior_variance(data, idx=idx)
        if model.X is None:  # if model is empty, second term vanishes
            return k_sum_x
        k_sum_X_x = self._raw_prior_covariance(InputData(model.X), data, idx=idx)
        v = scipy.linalg.solve_triangular(cholesky, k_sum_X_x, lower=True)
        var = k_sum_x - np.einsum("ij,ij->j", v, v).reshape(-1, 1)
        if self._within_model_normalize:
            var *= model.y_normalizer.scale_ ** 2
        return var

    def _raw_prior_variance(self, data: InputData, idx: int = None) -> np.ndarray:
        """Calculate the prior variance (=sum kernel) for a given test point in the
        normalized space of `idx`-th GP.

        Args:
            data: Input to be queried. `shape = (n_points, n_features)`
            idx: GP index in the stack. Counting happens from bottom (=first source GP)
                to top (=target GP) of the stack.
                Defaults to None, which corresponds to target GP.

        Returns:
            Prior variance. `shape = (n_points, 1)`
        """
        idx = self._idx_to_int(idx)
        model_idx, _ = self._idx_to_model_and_cholesky(idx)
        kernel_idx = model_idx.compute_kernel_diagonal(data)
        posterior_idx_minus_one = self._predict_posterior_variance(data, idx=idx - 1)
        if self._within_model_normalize and model_idx.X is not None:
            posterior_idx_minus_one /= model_idx.y_normalizer.scale_ ** 2
        prior_idx = kernel_idx + posterior_idx_minus_one
        return prior_idx

    def _idx_to_model_and_cholesky(self, idx: int = None) -> (GPBO, np.ndarray):
        """Return the `GPBO` model and Cholesky matrix corresponding to `idx`"""
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )
        idx = self._idx_to_int(idx)

        all_models = self.source_gps + [self.target_gp]
        all_cholesky = self._cholesky_source + [self._cholesky]
        model = all_models[idx]
        cholesky = all_cholesky[idx]
        return model, cholesky

    def _idx_to_int(self, idx: Union[int, None]) -> int:
        """Return the integer value of `idx` in case `idx=None`."""
        if idx is None:  # if None, it's set to the index of the target GP
            idx = len(self.source_gps)
        return idx

    def _get_posterior_cov_kernel(
        self, data: InputData, scale: float, idx: int = None
    ) -> Kern:
        """Return the sum kernel for training the stack. The posterior covariance of
        the previous GP, `idx-1`, is added to the kernel of the `idx`-th GP.

        Args:
            data: Input points at which the posterior covariance of the current stack will
                be evaluated.
            scale: Normalization scale of the y-values with respect to their standard
                deviation in the `idx`-th space. Relevant for normalized models.
                Set to `1.0` for unnormalized models.
            idx: GP index in the stack. Counting happens from bottom to top of the
                stack. Defaults to `None`, which corresponds to target GP.

        Returns:
            `GPy.Kern` object.
        """
        idx = self._idx_to_int(idx)
        model, _ = self._idx_to_model_and_cholesky(idx)

        posterior_covariance_matrix = self.predict_posterior_covariance(
            data, data, idx=idx - 1
        )
        if self._within_model_normalize:
            posterior_covariance_matrix /= scale ** 2
        posterior_cov_kernel = FixedKernel(
            input_dim=self.n_features,
            covariance_matrix=posterior_covariance_matrix,
        )

        return posterior_cov_kernel
