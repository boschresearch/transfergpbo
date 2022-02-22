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

from transfergpbo.models import HGP
from transfergpbo.models.utils import CrossTaskKernel


class WSGP(HGP):
    r"""Weighted Source GP, a GP-based transfer-learning algorithm.

    Weighted Source GP is a kernel method that models the difference between the target
    data and a weighted sum of source models. The coefficients of this weighted sum
    quantify the amount of correlation between source and target and are optimized
    during training along with the other hyperparameters by maximizing the likelihood
    function. Weighted Source GP neglects correlations between different source tasks.
    For one source and one target task, the kernel takes the form

    $$
        \begin{bmatrix}
        k((x, s), (x', s)) &  k((x, s), (x', t)) \\
        k((x, t), (x', s)) &  k((x, t), (x', t))
        \end{bmatrix}
        =
        \begin{bmatrix}
        (1+w)k_s & wk_s \\
        wk_s & wk_s + k_t
        \end{bmatrix}.
    $$

    Here, no weighing is performed since only once source task is present. Extension to
    multiple sources is carried out by maintaining the above kernel structure and
    setting the correlation between different source tasks to zero. For instance, for
    two source tasks, the kernel reads

    $$
        \begin{bmatrix}
        k((x, s_1), (x', s_1)) &  k((x, s_1), (x', s_2)) & k((x, s_1), (x', t)) \\
        k((x, s_2), (x', s_1)) &  k((x, s_2), (x', s_2)) & k((x, s_2), (x', t)) \\
        k((x, t), (x', s_1)) &  k((x, t), (x', s_2)) & k((x, t), (x', t))
        \end{bmatrix}
        =
        \begin{bmatrix}
        (1+w_1)k_{s_1} & 0              & w_1k_{s_1} \\
        0              & (1+w_2)k_{s_2} & w_2k_{s_2} \\
        w_1k_{s_1}     & w_2k_{s_2}     & w_1k_{s_1} + w_2k_{s_2} + k_t
        \end{bmatrix}.
    $$

    Weighted Source GP is a powerful but computationally expensive method since (i) it
    scales cubically with the total number of data points and (ii) the number of
    hyperparameters scales linearly with the number of tasks. The approach is
    particularly useful for the case when the hierarchical importance of the different
    source tasks is unknown beforehand.
    """

    def _get_kernel(self, num_features: int, num_source_tasks: int):
        source_kernels = []
        target_index = num_source_tasks
        task_index_dim = num_features
        # The kernel is decomposed as (domain + c_1 * cross) * c_2 * rbf. That
        # way the target kernel can choose how much of the signal correlated with
        # the target.
        # The default values are chosen so that (1 + c_1) * c_2 = 1 and we pick the
        # target correlation (c_1) such that the target covariance correlates with an
        # average of all source tasks. I.e. sum_source target_covariance = 1
        target_covariance = 1.0 / num_source_tasks

        for i in range(num_source_tasks):
            task_domain = CrossTaskKernel(
                task_indices=i,
                index_dim=task_index_dim,
                variance=1.0,
            )
            task_domain.variance.fix()
            target_correlation = CrossTaskKernel(
                task_indices=(i, target_index),
                index_dim=task_index_dim,
                variance=target_covariance,
            )
            domain_kernel = task_domain + target_correlation
            task_kernel = self._default_kernel.copy()
            # (1 + c_1) * c_2 = 1
            task_kernel.variance /= 1.0 + target_covariance
            source_kernels.append(task_kernel * domain_kernel)

        target_domain = CrossTaskKernel(
            task_indices=target_index, index_dim=task_index_dim
        )
        target_domain.variance.fix()
        target_kernel = self._default_kernel.copy() * target_domain

        return sum(source_kernels, target_kernel)
