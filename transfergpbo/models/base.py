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

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple
import numpy as np
from emukit.core.interfaces import IModel


@dataclass
class InputData:
    X: np.ndarray


@dataclass
class TaskData:
    X: np.ndarray
    Y: np.ndarray


class Model(ABC):
    """Abstract model class."""

    def __init__(self):
        """Initializes base model."""
        self._X = None
        self._y = None

    @property
    def X(self) -> np.ndarray:
        """Return input data."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Return target data."""
        return self._y

    @abstractmethod
    def meta_fit(self, metadata: Dict[Hashable, TaskData], **kwargs):
        """Train model on historical data.

        Parameters:
        -----------
        metadata
            Dictionary containing a numerical representation of the meta-data that can
            be used to meta-train a model for each task.
        """
        pass

    @abstractmethod
    def fit(self, data: TaskData, **kwargs):
        """Adjust model parameter to the observation on the new dataset.

        Parameters:
        -----------
        data: TaskData
            Observation data.
        """
        pass

    @abstractmethod
    def predict(self, data: InputData) -> (np.ndarray, np.ndarray):
        """Predict outcomes for a given array of input values.

        Parameters:
        -----------
        data: InputData
            Input data to predict on.

        Returns
        -------
        mu: shape = (n_points, 1)
            Predicted mean for every input
        cov: shape = (n_points, n_points) or (n_points, 1)
            Predicted (co-)variance for every input
        """
        pass


class WrapperBase(IModel):
    def __init__(self, model: Model):
        self._wrapped_model = model

    def __getattr__(self, item):
        return getattr(self._wrapped_model, item)

    @property
    def X(self):
        return self._wrapped_model.X

    @property
    def Y(self):
        return self._wrapped_model.y

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        return self._wrapped_model.fit(data=TaskData(X=X, Y=Y), optimize=False)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._wrapped_model.predict(InputData(X=X))

    def optimize(self) -> None:
        return self._wrapped_model.fit(data=TaskData(X=self.X, Y=self.Y), optimize=True)
