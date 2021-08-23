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


import joblib

from sedna.common.utils import get_func_spec

from ..base import BackendBase


class SklearnBackend(BackendBase):

	def __init__(self, estimator, **kwargs):
		super().__init__(estimator=estimator, **kwargs)

	def train(self, train_data, valid_data=None, **kwargs):
		pass

	def evaluate(self, valid_data, **kwargs):
		pass

	def predict(self, data, **kwargs):
		pass

	def save(self, model_url="", model_name=None):
		pass

	def load(self, model_url="", model_name=None, **kwargs):
		pass
