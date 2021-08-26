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

"""This script contains some common tools."""

import socket
from functools import wraps
from inspect import getfullargspec


def get_host_ip():
    """Get local ip address."""
    ip = '127.0.0.1'
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        pass
    finally:
        s.close()

    return ip


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance


def model_layer_flatten(weights):
    """like this:
    weights.shape=[(3, 3, 3, 64), (64,), (3, 3, 64, 32), (32,), (6272, 64),
        (64,), (64, 32), (32,), (32, 2), (2,)]
    flatten_weights=[(1728,), (64,), (18432,), (32,), (401408,), (64,),
        (2048,), (32,), (64,), (2,)]
    :param weights:
    :return:
    """
    flatten = [layer.reshape((-1)) for layer in weights]
    return flatten


def model_layer_reshape(flatten_weights, shapes):
    shaped_model = []
    for idx, flatten_layer in enumerate(flatten_weights):
        shaped_model.append(flatten_layer.reshape(shapes[idx]))
    return shaped_model


def get_func_spec(func, **kwargs):
    """

    :param func:
    :param kwargs:
    :return:
    """

    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs

    return {k: v for k, v in kwargs.items() if k in need_kw.args}

