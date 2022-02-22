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

import numpy as np

from transfergpbo.models import InputData, TaskData, Model, GPBO


class RGPE(Model):
    """Ranking-weighted Gaussian process ensemble implementation.

    Based on the paper "Scalable Meta-Learning for Bayesian Optimization" by Matthias
    Feurer, Benjamin Letham, and Eytan Bakshy (2018). The paper can be found
    [here](https://arxiv.org/pdf/1802.02219.pdf).
    """

    def __init__(
        self,
        n_samples: int = 256,
        n_sample_retries: int = 16,
        start: str = "random-random",
        **gpygp_kwargs: dict,
    ):
        """Initialize the ranking-weighted Gaussian process ensemble.

        Args:
            n_samples: Number of samples to calculate loss and model weights from. Must
                be larger than `0`.
            n_sample_retries: Number of retries before failing the sampling process
                during loss calculation (required for weight calculation). This prevents
                numerical instabilities from failing a fit call.
            start: Defines the initialization of the model before two unique target
                observations are available, since the `RGPE` only works under that
                condition. Must be one of "random-random", "random-mean",
                "mean-mean", "random-weighted", or "mean-weighted". For each
                variant, the part before the dash defines the sampling of the first
                point and the part after the dash defines the sampling of the second
                point. For "random", an observation is taken at a random point. For
                "mean", the mean of all base models (and the target model, if one
                target observation exists) is used. For "weighted", the base models
                are weighted based on the probability of producing the closest guess
                of the observed value.
            **gpygp_kwargs: Named arguments for initializing `GPBO` models.
        """

        if n_samples < 1:
            raise ValueError("The parameter n_samples must be larger than 0.")

        valid_starts = [
            "random-random",
            "random-mean",
            "mean-mean",
            "random-weighted",
            "mean-weighted",
        ]
        if start not in valid_starts:
            raise ValueError(
                f"The parameter start must be one of {','.join(valid_starts)}."
            )

        super().__init__()

        self._n_samples = n_samples
        self._n_sample_retries = n_sample_retries
        self._start = start
        self._gpygp_kwargs = gpygp_kwargs

        self._metadata = {}
        self._source_gps = {}
        self._source_gp_weights = {}

        self._target_model = GPBO(**copy.deepcopy(gpygp_kwargs))
        self._target_model_weight = 1

    @property
    def min_target_observations(self) -> int:
        """Return the minimum number of target observations before predict can
        be used. This differs depending on the `RGPE` variant, as described in
        `__init__`.

        Returns:
            Minimum number of target observations.
        """
        first, _, second = self._start.partition("-")

        if second == "random":
            return 2

        if first == "random":
            return 1

        return 0

    def meta_fit(self, metadata: Dict[Hashable, TaskData], **kwargs):
        """Train model on base task data. Fits a GP to each task and then
        calculates the weights for all models.

        Args:
            metadata: Numerical representation of metadata. Maps task UIDs (int) to
                their data classes. Each data class must have at least the variables
                `X` and `y`, where
                - `X`: `np.ndarray` of `float64`,
                    `shape = (n_observations_per_task, n_features)`
                - `y`: `np.ndarray` of `float64`, `shape = (n_observations_per_task, 1)`
        """

        if len(metadata) == 0:
            raise ValueError("Empty metadata given.")

        # check whether all base tasks have the same number of features
        n_features = [task_data.X.shape[1] for task_data in metadata.values()]
        if n_features.count(n_features[0]) < len(n_features):
            raise ValueError("Not all base tasks have the same number of features.")

        self._metadata = metadata

        # train model for each base task
        self._source_gps = {}
        for task_uid, task_data in metadata.items():
            model = GPBO(**copy.deepcopy(self._gpygp_kwargs))
            model.fit(task_data, optimize=True)
            self._source_gps[task_uid] = model

        self._calculate_weights()

    def fit(
        self,
        data: TaskData,
        optimize: bool = False,
    ):
        """Train model on target task data. Fits a GP to the task and then
        calculates the weights for all models.

        Args:
            data: Observation (input and output) data. Input shape
                `(n_points, n_features)`. Output shape `(n_points, 1)`.
            optimize: Run hyperparameter optimization.
        """

        if len(self._metadata) == 0:
            raise ValueError("No metadata is found. Forgot to run meta_fit?")

        # check whether the target task has the same number of features as a
        # base task
        n_features = next(iter(self._metadata.values())).X.shape[1]
        if data.X.shape[1] != n_features:
            raise ValueError(
                "The target task has a different number of features than the"
                " base tasks."
            )

        self._X = data.X
        self._y = data.Y

        # train model for target task if it will we used (when at least 1 target
        # task observation exists)
        if data.X.shape[0] >= 1:
            self._target_model.fit(data, optimize)

        self._calculate_weights()

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_test = data.X
        n_models = len(self._source_gp_weights)
        if self._target_model_weight > 0:
            n_models += 1
        n_points = X_test.shape[0]
        means = np.empty((n_models, n_points, 1))
        if return_full:
            vars_ = np.empty((n_models, n_points, n_points))
        else:
            vars_ = np.empty((n_models, n_points, 1))
        weights = np.empty((n_models, 1, 1))

        # calculate means and vars of base models
        for i, (task_uid, weight) in enumerate(self._source_gp_weights.items()):
            means[i], vars_[i] = self._source_gps[task_uid].predict(
                data, return_full=return_full, with_noise=with_noise
            )
            weights[i] = weight

        # calculate mean and var of target model, if necessary
        if self._target_model_weight > 0:
            means[-1], vars_[-1] = self._target_model.predict(
                data, return_full=return_full, with_noise=with_noise
            )
            weights[-1] = self._target_model_weight

        # calculate weighted mean and var
        mean = np.sum(weights * means, axis=0)
        var = np.sum(weights ** 2 * vars_, axis=0)

        return mean, var

    def _calculate_weights(self):
        """Calculate the weights of the target model and all base models.

        Calculates the weights according to the given start method if less than
        two unique target task observations exist. Otherwise, calculates the
        loss of n_samples samples for the target model and all base models.
        Discards all models which have a median loss higher than the 95th
        percentile of the target model loss. Assigns a weight to the remaining
        models based on the proportion of samples where a model has the lowest loss.
        """

        if self._X is None:
            # no target task observations
            self._calculate_weights_with_no_observations()
            return

        if np.unique(self._X, axis=0).shape[0] == 1:
            # one unique target task observation
            self._calculate_weights_with_one_observation()
            return

        # at least two unique target task observations
        # compute loss of the target model
        target_loss = self._target_loss()

        # compute loss of all base models and
        base_losses = np.empty((len(self._source_gps), self._n_samples))
        for i, task_uid in enumerate(self._source_gps):
            base_losses[i] = self._base_loss(task_uid)

        # discard base models with a mean loss higher than the 95th percentile
        # of the target model loss by setting their loss to an impossible high
        # value
        threshold = np.percentile(target_loss, 95)
        discard = np.mean(base_losses, axis=1) > threshold
        base_losses[discard] = self._X.shape[0] ** 2

        # get base models with lowest loss for each sample
        lowest_base_loss = np.min(base_losses, axis=0)
        is_best_source_gp = base_losses == lowest_base_loss

        # break ties randomly to get the best model of each sample
        # if the target model is at least as good as the best base model, it is
        # chosen instead
        tie_break = np.random.random(base_losses.shape)
        best = np.argmax(is_best_source_gp * tie_break, axis=0)
        best[target_loss <= lowest_base_loss] = len(self._source_gps)

        # compute weight as proportion of samples where model is best
        occurences = np.bincount(best, minlength=len(self._source_gps) + 1)
        weights = occurences / self._n_samples

        # store target model weight and assign weights to base task uids
        self._source_gp_weights = dict(zip(self._source_gps, weights[:-1]))
        self._target_model_weight = weights[-1]

    def _calculate_weights_with_no_observations(self):
        """Calculate weights according to the given start method when no target
        task observations exist.
        """

        first, _, _ = self._start.partition("-")

        if first == "random":
            # do nothing, predict should not yet be used
            return

        if first == "mean":
            # assign equal weights to all base models
            weight = 1 / len(self._source_gps)
            self._source_gp_weights = {
                task_uid: weight for task_uid in self._source_gps
            }
            self._target_model_weight = 0
            return

        raise RuntimeError(f"Predict called without observations, first = {first}")

    def _calculate_weights_with_one_observation(self):
        """Calculate weights according to the given start method when only one
        unique target task observation is available.
        """

        _, _, second = self._start.partition("-")

        if second == "random":
            # do nothing, predict should not be used yet
            return

        if second == "mean":
            # assign equal weights to all base models and the target model
            weight = 1 / (len(self._source_gps) + 1)
            self._source_gp_weights = {
                task_uid: weight for task_uid in self._source_gps
            }
            self._target_model_weight = weight
            return

        if second == "weighted":
            # get unique observed point
            X, indices = np.unique(self._X, axis=0, return_index=True)

            # draw _n_samples for each unique observed point from each
            # base model
            all_samples = np.empty((len(self._source_gps), self._n_samples))
            for i, task_uid in enumerate(self._source_gps):
                model = self._source_gps[task_uid]
                samples = model.sample(
                    InputData(X), size=self._n_samples, with_noise=True
                )
                all_samples[i] = samples

            # compare drawn samples to observed values
            y = self._y[indices]
            diff = np.abs(all_samples - y)

            # get base model with lowest absolute difference for each sample
            best = np.argmin(diff, axis=0)

            # compute weight as proportion of samples where a base model is best
            occurences = np.bincount(best, minlength=len(self._source_gps))
            weights = occurences / self._n_samples
            self._source_gp_weights = dict(zip(self._source_gps, weights))
            self._target_model_weight = 0
            return

        raise RuntimeError(
            f"Weight calculation with one observation, second = {second}"
        )

    def _base_loss(self, task_uid: int) -> np.ndarray:
        """Calculate the loss of the base model for the given task.

        Draws `n_samples` samples from the base model at the observed `X` of the
        target task without duplicates. Compares each value against each other
        value. Compares the relative rankings against the relative rankings of
        the observed y of the target task. Returns the number of misrankings per
        sample as the loss.

        Args:
            task_uid: UID of the task.

        Returns:
            Number of misrankings per sample. `shape = (n_samples, )`
        """

        # get the trained model for the base task
        model = self._source_gps[task_uid]

        # get points observed on the target task with duplicates removed
        X, indices = np.unique(self._X, axis=0, return_index=True)
        y = self._y[indices]

        # draw n_samples samples from the trained model
        # compare every value to every other value for each sample
        # retry up to a limit to prevent numerical instabilities from failing
        # the loss calculation
        for _ in range(self._n_sample_retries):
            try:
                samples = model.sample(
                    InputData(X), size=self._n_samples, with_noise=True
                )
                break
            except (ValueError, np.linalg.LinAlgError) as exception:
                last_exception = exception
        else:
            raise last_exception
        sample_comps = samples[:, np.newaxis, :] < samples

        # calculate loss with sample comparisons
        return self._compute_loss(sample_comps, y)

    def _target_loss(self) -> np.ndarray:
        """Calculate the loss of the target model.

        Draws `n_samples` samples from the target model at the observed `X` of the
        target task without duplicates. Compares each value against each other value.
        Compares the relative rankings against the relative rankings of the observed `y`
        of the target task. Returns the number of misrankings per sample as the loss.

        Returns:
            Number of misrankings per sample. `shape = (n_samples, )`
        """

        # get a copy of the trained model for the target task, since its
        # training data is altered during loss calculation
        model = copy.deepcopy(self._target_model)

        # get points observed on the target task with duplicates removed
        all_X, indices = np.unique(self._X, axis=0, return_index=True)
        all_y = self._y[indices]

        # collect sample comparisons for each leave-one-out model
        n_points = all_X.shape[0]
        sample_comps = np.empty((n_points, n_points, self._n_samples), dtype=bool)
        for i in range(n_points):
            # leave i-th element out
            # train model without optimization
            X = np.delete(all_X, i, axis=0)
            y = np.delete(all_y, i, axis=0)
            model.fit(TaskData(X, y), optimize=False)

            # draw n_samples samples from the trained model
            # compare the i-th value to every other value for each sample
            # retry up to a limit to prevent numerical instabilities from failing
            # the loss calculation
            for _ in range(self._n_sample_retries):
                try:
                    samples = model.sample(
                        InputData(all_X), size=self._n_samples, with_noise=True
                    )
                    break
                except (ValueError, np.linalg.LinAlgError) as exception:
                    last_exception = exception
            else:
                raise last_exception
            sample_comps[i] = samples[i, :] < samples

        # calculate loss with sample comparisons
        return self._compute_loss(sample_comps, all_y)

    def _compute_loss(self, sample_comps: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the loss of the given sample comparisons.

        The sample comparisons are assumed to be generated by drawing `n_samples`
        samples from a model at certain points and then comparing each value
        against each other value. This method compares the given relative
        rankings against the relative rankings of the given observed `y` at the
        same points of the target task. Returns the number of misrankings per
        sample as the loss.

        Args:
            sample_comps: Relative rankings of values of n_samples samples drawn from a
                model. `shape = (n_points, n_points, n_samples)`
            y: Observed values of the target task at the points at which samples
                were drawn. `shape = (n_points, )`

        Returns:
            Number of misrankings per sample. `shape = (n_samples, )`
        """

        # compare every target observation to every other target observation
        # repeat the comparison for every sample for XORing
        target_comps = np.tile(y[:, np.newaxis, :] < y, self._n_samples)

        # XOR sample and target comparisons
        # calculate loss per sample (count of different comparison results)
        return np.sum(sample_comps ^ target_comps, axis=(1, 0))
