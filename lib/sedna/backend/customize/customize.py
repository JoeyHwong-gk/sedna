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

import os.path

from sedna.common.file_ops import FileOps
from sedna.common.utils import get_func_spec

from ..base import BackendBase


class CustomizeBackend(BackendBase):
	"""Customize Framework Backend Class"""

	def __init__(self, estimator, **kwargs):
		super().__init__(estimator=estimator, **kwargs)
		self.model_name = ""

	def train(self, *args, **kwargs):
		"""Train model."""
		if callable(self.estimator):
			varkw = get_func_spec(self.estimator, **kwargs)
			self.estimator = self.estimator(**varkw)
		fit_method = getattr(self.estimator, "fit", self.estimator.train)
		varkw = get_func_spec(fit_method, **kwargs)
		return fit_method(*args, **varkw)

	def predict(self, *args, **kwargs):
		"""Inference model."""
		varkw = get_func_spec(self.estimator.predict, **kwargs)
		return self.estimator.predict(*args, **varkw)

	def predict_proba(self, *args, **kwargs):
		"""Compute probabilities of possible outcomes for samples in X."""
		varkw = get_func_spec(self.estimator.predict_proba, **kwargs)
		return self.estimator.predict_proba(*args, **varkw)

	def evaluate(self, *args, **kwargs):
		"""evaluate model."""
		varkw = get_func_spec(self.estimator.evaluate, **kwargs)
		return self.estimator.evaluate(*args, **varkw)

	def save(self, model_url="", model_name=None):
		mname = model_name or self.model_name
		if os.path.isfile(self.model_save_path):
			self.model_save_path, mname = os.path.split(self.model_save_path)

		FileOps.clean_folder([self.model_save_path], clean=False)
		model_path = FileOps.join_path(self.model_save_path, mname)
		self.estimator.save(model_path)
		if model_url and FileOps.exists(model_path):
			FileOps.upload(model_path, model_url)
			model_path = model_url
		return model_path

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

	def load(self, model_url="", model_name=None, **kwargs):
		mname = model_name or self.model_name
		if callable(self.estimator):
			varkw = get_func_spec(self.estimator, **kwargs)
			self.estimator = self.estimator(**varkw)
		if os.path.isfile(self.model_save_path):
			self.model_save_path, mname = os.path.split(self.model_save_path)
		model_path = FileOps.join_path(self.model_save_path, mname)
		if model_url:
			model_path = FileOps.download(model_url, model_path)
		self.has_load = True
		if not (hasattr(self.estimator, "load")
		        and os.path.exists(model_path)):
			return
		return self.estimator.load(model_url=model_path)
