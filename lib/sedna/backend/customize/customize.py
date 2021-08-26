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

from sedna.common.file_ops import FileOps
from sedna.common.utils import get_func_spec

from ..base import BackendBase


class CustomizeBackend(BackendBase):
    """Customize Framework Backend Class"""

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)
        self.framework = "customize"
        self.model_suffix = ".joblib"

    def train(self, train_data, **kwargs):
        fit_method = getattr(self.estimator, "fit", self.estimator.train)
        varkw = get_func_spec(fit_method, **kwargs)
        return fit_method(train_data, **varkw)

    def _predict(self, data, **kwargs):
        hyperparams = get_func_spec(self.estimator.predict, **kwargs)
        return self.estimator.predict(data, **hyperparams)

    def save(self, model_url=None, **kwargs):
        model_url = self.get_model_absolute_path(
            model_url=model_url,
            model_name=kwargs.get("model_name", None)
        )
        kwargs["model_url"] = model_url
        if hasattr(self.estimator, "save"):
            varkw = get_func_spec(self.estimator.save, **kwargs)
            self.estimator.save(**varkw)
            return model_url

        return FileOps.dump(self.estimator, model_url)

    def load(self, model_url=None, **kwargs):
        model_url = self.get_model_absolute_path(
            model_url=model_url,
            model_name=kwargs.get("model_name", None)
        )
        kwargs["model_url"] = model_url
        if hasattr(self.estimator, "load"):
            varkw = get_func_spec(self.estimator.load, **kwargs)
            return self.estimator.load(**varkw)
        self.estimator = FileOps.load(model_url)
