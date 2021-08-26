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


import keras
import tensorflow as tf
import numpy as np

from sedna.common.utils import get_func_spec
from sedna.common.file_ops import FileOps

from .tensorflow import TFBackend


class KerasBackend(TFBackend):
    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)
        if issubclass(type(estimator), keras.models.Model):
            self.framework = "keras"
        elif issubclass(type(estimator), tf.keras.models.Model):
            self.framework = "tf-keras"
        else:
            raise ValueError('Compiled keras model needs to be provided '
                             '(keras.models/tensorflow.keras.models). '
                             'Type provided' + str(type(estimator)))

    def set_session(self):
        if self.framework == "tf-keras":
            from keras.backend.tensorflow_backend import set_session
        else:
            from tensorflow.python.keras.backend import set_session
        return set_session(self.sess)

    def train(self, train_data, valid_data=None, **kwargs):
        hyperparams = get_func_spec(self.estimator.fit, **kwargs)

        x, y = train_data.x, train_data.y
        self.set_session()
        if valid_data:
            x1, y1 = valid_data.x, valid_data.y
            hyperparams["validation_data"] = (x1, y1)
        history = self.estimator.fit(x, y, **hyperparams)
        self.result = history.history
        self.has_load = True

    def _predict(self, data, **kwargs):
        hyperparams = get_func_spec(self.estimator.predict, **kwargs)
        self.set_session()
        return self.estimator.predict(data, **hyperparams)

    def evaluate(self, data, with_result=False, **kwargs):
        if not self.has_load:
            self.load(**kwargs)
        hyperparams = get_func_spec(self.estimator.evaluate, **kwargs)
        x, y = data.x, data.y
        self.set_session()
        metrics = self.estimator.evaluate(x, y, **hyperparams)
        names = self.estimator.metrics_names
        dict_metrics = {}
        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics

        if with_result:
            y_pred = self.predict(x, **kwargs)
            dict_metrics["_pred"] = y_pred
        return dict_metrics

    def save(self, model_url="", model_name=None):
        model_url = self.get_model_absolute_path(
            model_url=model_url, model_name=model_name
        )
        self.estimator.save(model_url)
        return model_url

    def load(self, model_url="", model_name=None, **kwargs):
        if not model_url:
            model_url = self.get_model_absolute_path(
                model_url=model_url,
                model_name=kwargs.get("model_name", None)
            )
        if not FileOps.exists(model_url):
            model_url = FileOps.download(model_url)
        tmp = {
            k: v for k, v in kwargs.items() if k in (
                "custom_objects", "compile", "options")
        }
        if self.framework == "keras":
            self.estimator = keras.models.load_model(model_url, **tmp)
        else:
            self.estimator = tf.keras.models.load_model(model_url, **tmp)
        self.has_load = True

    def get_weights(self):
        return list(map(lambda x: x.tolist(), self.estimator.get_weights()))

    def set_weights(self, weights):
        weights = [np.array(x) for x in weights]
        self.estimator.set_weights(weights)
