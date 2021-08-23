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

import os
import abc

from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource


class BackendBase(abc.ABC):
	"""ML Framework Backend base Class"""

	def __init__(self,
	             estimator,
	             use_cuda: bool = False,
	             model_save_path: str = "",
	             model_name: str = "",
	             model_save_url: str = "",
	             **kwargs):
		self.framework = ""
		self.estimator = estimator
		self.use_cuda = use_cuda
		self.model_save_url = model_save_url
		self.model_save_path = model_save_path or "/tmp"
		self.default_name = model_name
		self.has_load = False
		self.initial_param = kwargs
		self.result = None

	@abc.abstractmethod
	def train(self,
	          train_data: BaseDataSource,
	          valid_data: BaseDataSource = None,
	          **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def predict(self, data, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def evaluate(self, valid_data: BaseDataSource, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def save(self, model_url="", model_name=None):
		raise NotImplementedError

	@abc.abstractmethod
	def load(self, model_url="", model_name=None, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def set_weights(self, weights):
		"""Set weight with memory tensor."""
		raise NotImplementedError

	@abc.abstractmethod
	def get_weights(self):
		"""Get the weights."""
		raise NotImplementedError

	def model_info(self, model, relpath=None, result=None):
		_, _type = os.path.splitext(model)
		if relpath:
			_url = FileOps.remove_path_prefix(model, relpath)
		else:
			_url = model
		results = [{
			"format": _type.lstrip("."),
			"url": _url,
			"metrics": result
		}]
		return results
