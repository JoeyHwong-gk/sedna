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

import tensorflow as tf

from sedna.common.file_ops import FileOps
from sedna.common.utils import get_func_spec

from ..base import BackendBase


if hasattr(tf, "compat"):
    # version 2.0 tf
    ConfigProto = tf.compat.v1.ConfigProto
    Session = tf.compat.v1.Session
    reset_default_graph = tf.compat.v1.reset_default_graph
    get_default_graph = tf.compat.v1.get_default_graph
else:
    # version 1
    ConfigProto = tf.ConfigProto
    Session = tf.Session
    reset_default_graph = tf.reset_default_graph
    get_default_graph = tf.get_default_graph


class TFBackend(BackendBase):
    """
    Wrapper class for importing tensorflow models.

    Parameters
    ==========
        estimator : tf.estimator


    Examples
    ========
    >>> estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config,
            params={"config": config})
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator, **kwargs)
        self.framework = "tensorflow"

        sess_config = (self._init_gpu_session_config()
                       if self.use_cuda else self._init_cpu_session_config())
        self.graph = get_default_graph()

        with self.graph.as_default():
            self.sess = Session(config=sess_config)

    def train(self, train_data, valid_data=None, **kwargs):
        hyperparams = get_func_spec(self.estimator.train, **kwargs)
        return self.estimator.train(
            train_data=train_data,
            valid_data=valid_data,
            **hyperparams
        )

    def predict(self, data, **kwargs):
        hyperparams = get_func_spec(self.estimator.predict, **kwargs)
        return self.estimator.predict(data=data, **hyperparams)

    def evaluate(self, valid_data, **kwargs):
        hyperparams = get_func_spec(self.estimator.evaluate, **kwargs)
        return self.estimator.evaluate(valid_data=valid_data, **hyperparams)

    def save(self, model_url="", model_name=None):
        pass

    def load(self, model_url="", model_name=None, **kwargs):
        pass

    @staticmethod
    def _init_cpu_session_config():
        sess_config = ConfigProto(allow_soft_placement=True)
        return sess_config

    @staticmethod
    def _init_gpu_session_config():
        sess_config = ConfigProto(
            log_device_placement=True, allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True
        return sess_config

    def model_info(self, model, relpath=None, result=None):
        ckpt = os.path.dirname(model)
        _, _type = os.path.splitext(model)
        if relpath:
            _url = FileOps.remove_path_prefix(model, relpath)
            ckpt_url = FileOps.remove_path_prefix(ckpt, relpath)
        else:
            _url = model
            ckpt_url = ckpt
        _type = _type.lstrip(".").lower()
        results = [{
                "format": _type,
                "url": _url,
                "metrics": result
            }]
        if _type == "pb":  # report ckpt path when model save as pb file
            results.append({
                "format": "ckpt",
                "url": ckpt_url,
                "metrics": result
            })
        return results
