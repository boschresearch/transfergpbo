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

from typing import Tuple, Union

import numpy as np
from transfergpbo.models import (
    InputData,
    TaskData,
    GPBO,
    MHGP,
    SHGP,
)
from transfergpbo.models.utils import compute_alpha


class BHGP(MHGP):
    r"""Boosted Hierarchical GP, a GP-based transfer-learning algorithm.

    Similarly to SHGP, BHGP improves over MHGP
    by propagating the uncertainty through the stack of GPs. In contrast to Uncertainty
    Inheritance, BHGP is _not_ a Bayesian technique and propagates
    uncertainty heuristically. For generic applications we recommend the use of
    SHGP over BoostedGP. The design and API are the same as those
    of MHGP.

    The key idea of BHGP is to train a target GP with the prior mean function
    a sample of the source GP. The posterior distribution of the target GP
    is averaged over all the samples of the source GP. The resulting sample-averaged
    distribution is also a GP with the mean, $m_t(x)$, and covariance, $K_t(x, x')$,
    given by

    $$
        m_t(x) = m_\mathrm{MHGP}(x),
    $$

    $$
        K_t(x,x') = K_\mathrm{MHGP}(x, x') + \Sigma_*^\mathrm{boost}(x, x'),
    $$

    where $\Sigma_*^\mathrm{boost}(x, x')$ is the additional boosting uncertainty
    originating from the uncertainty of the source GP.

    In the presence of multiple source tasks, this implementation of BHGP
    adds a boosting uncertainty for the prediction of every task (except the very
    bottom source task).

    BHGP has the sample computation complexity as SHGP:

    - BHGP scales as $\mathcal{O}(N_tN_s^2)$ for training and
        $\mathcal{O}(N_s^2)$ for inference, where $N_s(N_t)$ is the number of source
        (target) points.

    - BHGP scales as $\mathcal{O}(N_tN_s)$ for training and $\mathcal{O}(N_s)$
        for inference.
    """

    def __init__(self, n_features, within_model_normalize=True):
        super().__init__(n_features, within_model_normalize=within_model_normalize)
        self._posterior_cov = None  # posterior cov of source evaluated at target points
        self._posterior_cov_source = []

    def _update_meta_data(self, *gps: GPBO):
        super()._update_meta_data(*gps)
        self._cache_source_quantities()

    def _cache_source_quantities(self):
        """Compute and cache the Cholesky decomposition of the source models"""
        if not self._posterior_cov_source:
            # the posterior cov of the first source model is not needed
            self._posterior_cov_source.append(None)
        for i, model in enumerate(self.source_gps[1:], start=1):
            try:
                self._posterior_cov_source[i]
            except IndexError:
                X = InputData(model.X)
                pcs = self.predict_posterior_covariance(X, X, idx=i - 1)
                self._posterior_cov_source.append(pcs)

    def fit(
        self,
        data: TaskData,
        optimize: bool = False,
    ):
        super().fit(data, optimize=optimize)
        idx = self._idx_to_int(None)
        X = InputData(data.X)
        self._posterior_cov = self.predict_posterior_covariance(X, X, idx - 1)

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        return SHGP.predict(self, data, return_full, with_noise)

    def predict_posterior_mean(self, data: InputData, idx: int = None) -> np.ndarray:
        return super().predict_posterior_mean(data, idx)

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
        model = self._idx_to_model(idx)

        cov_MHGP = model.predict_posterior_covariance(x1, x2)
        ks_x1_x2 = self.predict_posterior_covariance(x1, x2, idx=idx - 1)
        if model.X is None:
            return cov_MHGP + ks_x1_x2

        X = InputData(model.X)
        alpha_x1_X = compute_alpha(model, x1).T
        ks_X_X = self._idx_to_posterior(idx=idx)
        ks_x1_X = self.predict_posterior_covariance(x1, X, idx=idx - 1)

        if np.array_equal(x1.X, x2.X):
            alpha_X_x2 = alpha_x1_X.T
            ks_X_x2 = ks_x1_X.T
        else:
            alpha_X_x2 = compute_alpha(model, x2)
            ks_X_x2 = self.predict_posterior_covariance(X, x2, idx=idx - 1)

        kse_X_X = np.linalg.multi_dot([alpha_x1_X, ks_X_X, alpha_X_x2])
        kse_x1_X = ks_x1_X.dot(alpha_X_x2)
        kse_X_x2 = alpha_x1_X.dot(ks_X_x2)

        cov = cov_MHGP + ks_x1_x2 + kse_X_X - kse_x1_X - kse_X_x2
        return cov

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
        model = self._idx_to_model(idx)
        x = data
        var_MHGP = model.predict(x)[1]
        ks_x = self._predict_posterior_variance(x, idx=idx - 1)
        if model.X is None:
            return var_MHGP + ks_x
        alpha_X_x = compute_alpha(model, x)
        alpha_x_X = alpha_X_x.T
        X = InputData(model.X)
        ks_X_X = self._idx_to_posterior(idx=idx)
        ks_x_X = self.predict_posterior_covariance(x, X, idx=idx - 1)
        ks_X_x = ks_x_X.T

        kse_X_X = np.einsum("ij,jk,ki->i", alpha_x_X, ks_X_X, alpha_X_x).reshape(-1, 1)
        kse_X_x = np.einsum("ij,ji->i", alpha_x_X, ks_X_x).reshape(-1, 1)

        var = var_MHGP + ks_x + kse_X_X - 2 * kse_X_x
        return var

    def _idx_to_model(self, idx: int = None) -> GPBO:
        """Return the `GPBO` model corresponding to `idx`"""
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )
        idx = self._idx_to_int(idx)
        all_models = self.source_gps + [self.target_gp]
        model = all_models[idx]
        return model

    def _idx_to_posterior(self, idx: int = None) -> (GPBO, np.ndarray):
        """Return the posterior matrix corresponding to `idx`"""
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )
        idx = self._idx_to_int(idx)

        all_ps = self._posterior_cov_source + [self._posterior_cov]
        ps = all_ps[idx]
        return ps

    def _idx_to_int(self, idx: Union[int, None]) -> int:
        """Return the integer value of `idx` in case `idx=None`."""
        if idx is None:  # if None, it's set to the index of the target GP
            idx = len(self.source_gps)
        return idx
