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


import copy

from sedna.common.utils import get_func_spec
from sedna.common.file_ops import FileOps

from ..base import BackendBase


class TorchBackend(BackendBase):

	def __init__(self, estimator, **kwargs):
		super().__init__(estimator=estimator, **kwargs)

	def train(self, train_data, valid_data=None, **kwargs):
		hyperparams = get_func_spec(self.estimator.fit, **kwargs)
		x, y = train_data.x, train_data.y
		self.estimator.set_params(**hyperparams)
		self.estimator.fit_loop(x, y)

	def evaluate(self, valid_data, **kwargs):
		pass

	def predict(self, data, **kwargs):
		return self.estimator.predict(data)

	def save(self, model_url="", model_name=None):
		if model_name is None:
			file = self.default_name if self.default_name else self.framework
			model_name = f"{file}.pt"

		full_path = FileOps.join_path(model_url, model_name)
		f_optimizer = None
		f_history = None
		# if optimizer_filename is not None:
		# 	f_optimizer = super().get_model_absolute_path(optimizer_filename)
		# if history_filename is not None:
		# 	f_history = super().get_model_absolute_path(history_filename)
		self.estimator.save_params(
			f_params=full_path, f_optimizer=f_optimizer, f_history=f_history)

	def load(self, model_url="", model_name=None, **kwargs):
		pass

	def get_weights(self):
		if self.use_cuda:
			module = copy.deepcopy(self.estimator.module_).cpu()

		else:
			module = self.estimator.module_
		return list(module.parameters())

	def set_weights(self, weights):
		pass
