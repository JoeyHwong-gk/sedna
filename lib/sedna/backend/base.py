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
from sedna.common.utils import get_func_spec
from sedna.datasources import BaseDataSource


class BackendBase(abc.ABC):
    """
    Base class for all ML backend.
    """

    def __init__(self,
                 estimator,
                 use_cuda: bool = False,
                 model_name: str = "",
                 model_save_url: str = "",
                 **kwargs):
        self.framework = os.getenv('BACKEND_TYPE', 'senda').lower()
        self.model_suffix = ".pkl"
        self.estimator = estimator
        self.use_cuda = use_cuda
        self.model_save_url = model_save_url
        self.default_name = model_name
        self.has_load = False
        self.initial_param = kwargs
        self.result = None
        self.result_transform = kwargs.get("transform", None)
        if callable(self.estimator):
            varkw = get_func_spec(self.estimator, **kwargs)
            self.estimator = self.estimator(**varkw)

    def get_model_absolute_path(self, model_url="", model_name=""):
        if not model_name:
            file = self.default_name or f"{self.framework}_model"
            model_name = f'{file}{self.model_suffix}'
        if not model_url:
            model_url = self.model_save_url
        if os.path.isfile(model_url):
            model_url, model_name = os.path.split(model_url)
        if not (os.path.isfile(model_url) or
                str(model_url).endswith(self.model_suffix)):
            model_url = FileOps.join_path(model_url, model_name)
        return model_url

    @abc.abstractmethod
    def train(self,
              train_data: BaseDataSource,
              valid_data: BaseDataSource = None, **kwargs):
        """
        Fits current model with provided training data.

        Parameters
        ----------
        train_data: BaseDataSource
            datasource use for train, see
            `sedna.datasources.BaseDataSource` for more detail.
        valid_data:  BaseDataSource
            datasource use for evaluation, see
            `sedna.datasources.BaseDataSource` for more detail.
        kwargs: Dict
            Dictionary of model-specific arguments for fitting, \
            Like: `early_stopping_rounds` in Xgboost.XGBClassifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, data, **kwargs):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        """
         Perform prediction for a batch of inputs.

        Parameters
        ----------
        data: Data structure as expected by the model
            Samples with shape as expected by the model.
        kwargs: Dict
            Dictionary of model-specific arguments for predict, \
            Like: `ntree_limit` in Xgboost.XGBClassifier.
        """
        if not self.has_load:
            self.load(**kwargs)
        pred = self.predict(data, **kwargs)
        if callable(self.result_transform):
            varkw = get_func_spec(self.result_transform, **kwargs)
            pred = self.result_transform(pred, **varkw)
        return pred

    def evaluate(self, valid_data, with_result=False,
                 metric_func=None, **kwargs):
        """
        Evaluates model given the test dataset.
        Multiple evaluation metrics are returned in a dictionary

        Parameters
        ----------
        valid_data:  BaseDataSource
            datasource use for evaluation, see
            `sedna.datasources.BaseDataSource` for more detail.
        with_result: bool
        metric_func: functional
        kwargs: Dict
            Dictionary of model-specific arguments for evaluation, \
            Like: `metric_name` in Xgboost.XGBClassifier.
        """

        if not self.has_load:
            self.load(**kwargs)
        x, y = valid_data.x, valid_data.y
        metrics = dict()
        if hasattr(self.estimator, "evaluate"):
            hyperparams = get_func_spec(self.estimator.evaluate, **kwargs)
            metrics = self.estimator.evaluate(x, y, **hyperparams)
        if not metrics or with_result:
            pred = self.predict(x, **kwargs)
            metrics["_pred"] = pred
            if callable(metric_func):
                m_name = getattr(metric_func, '__name__',
                                 f"{self.framework}_eval")
                metrics_param = get_func_spec(**kwargs)
                metrics[m_name] = metric_func(y, pred, **metrics_param)
        return metrics

    @abc.abstractmethod
    def save(self, model_url="", model_name=None):
        """
        Save a model to file in the format specific to the backend framework.

        Parameters
        ----------
        model_name: str
            Name of the file where to store the model.
        model_url: str
            Path of the folder where to store the model.

        Returns
        -------
        filename
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, model_url="", **kwargs):
        """
        Load model from provided filename

        Parameters
        ----------
        model_url: str
            Path of the folder where to store the model.
        kwargs: Dict
            Dictionary of model-specific arguments for initial, \
            Like: `lr` in Xgboost.XGBClassifier.
        """
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
