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

from sedna.common.utils import get_func_spec
from sedna.common.file_ops import FileOps

from ..base import BackendBase


class SklearnBackend(BackendBase):

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)
        self.framework = "scikit-learn"
        self.model_suffix = ".joblib"

    def train(self, train_data, valid_data=None, **kwargs):
        if valid_data:
            x1, y1 = valid_data.x, valid_data.y
            kwargs["eval_set"] = [(x1, y1), ]
        hyperparams = get_func_spec(self.estimator.fit, **kwargs)
        x, y = train_data.x, train_data.y
        self.result = self.estimator.fit(x, y, **hyperparams)
        self.has_load = True

    def _predict(self, data, **kwargs):
        hyperparams = get_func_spec(self.estimator.predict, **kwargs)
        return self.estimator.predict(data, **hyperparams)

    def save(self, model_url="", model_name=None):
        model_url = self.get_model_absolute_path(
            model_url=model_url, model_name=model_name
        )
        return FileOps.dump(self.estimator, model_url)

    def load(self, model_url="", **kwargs):
        if not model_url:
            model_url = self.get_model_absolute_path(
                model_url=model_url,
                model_name=kwargs.get("model_name", None)
            )
        self.estimator = FileOps.load(model_url)
