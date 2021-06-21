# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregation algorithms"""

import abc
from copy import deepcopy

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('FedAvg',)


class BaseAggregation(metaclass=abc.ABCMeta):
    """Abstract class of aggregator"""

    def __init__(self):
        self.total_size = 0
        self.weights = None

    @abc.abstractmethod
    def aggregate(self, weights, size=0):
        """
        Some algorithms can be aggregated in sequence,
        but some can be calculated only after all aggregated data is uploaded.
        therefore, this abstractmethod should consider that all weights are
        uploaded.
        :param weights: weights received from node
        :param size: numbers of sample in each loop
        :return: final weights
        """


@ClassFactory.register(ClassType.FL_AGG)
class FedAvg(BaseAggregation, abc.ABC):
    """
    Federated averaging algorithm : Calculate the average weight
    according to the number of samples
    """

    def aggregate(self, weights, size=0):

        total_sample = self.total_size + size
        if not total_sample:
            return self.weights
        updates = []
        for inx, weight in enumerate(weights):
            old_weight = self.weights[inx]
            row_weight = ((np.array(weight) - old_weight) *
                          (size / total_sample) + old_weight)
            updates.append(row_weight)
        self.weights = deepcopy(updates)
        return updates
