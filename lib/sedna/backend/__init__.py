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

"""Framework Backend class."""

import os

from sedna.common.config import BaseConfig
from sedna.common.constant import MLFramework


def set_backend(estimator=None, config=None, **kwargs):
    """Create Trainer clss."""
    if estimator is None:
        return
    if config is None:
        config = BaseConfig()
    use_cuda = False
    backend_type = os.getenv(
        'BACKEND_TYPE', config.get("backend_type", "UNKNOWN")
    )
    backend_type = str(backend_type).lower()
    device_category = os.getenv(
        'DEVICE_CATEGORY', config.get("device_category", "CPU")
    )
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
        use_cuda = True
    else:
        os.environ['DEVICE_CATEGORY'] = device_category

    expect_attr_list = (
        "train", "predict", "load"
    )
    if all(map(lambda x: hasattr(estimator, x), expect_attr_list)):
        backend_type = "customize"
    if backend_type == MLFramework.TENSORFLOW.value:
        from sedna.backend.tensorflow.tensorflow import TFBackend as REGISTER
    elif backend_type == MLFramework.KERAS.value:
        from sedna.backend.tensorflow.keras import KerasBackend as REGISTER
    elif backend_type == MLFramework.SKLEARN.value:
        from sedna.backend.scikitlearn import SklearnBackend as REGISTER
    elif backend_type == MLFramework.TORCH.value:
        from sedna.backend.torch import TorchBackend as REGISTER
    else:
        from sedna.backend.customize import CustomizeBackend as REGISTER
        
    model_save_url = config.get("model_url")
    model_save_name = config.get("model_name")
    return REGISTER(
        estimator=estimator, use_cuda=use_cuda,
        model_name=model_save_name,
        model_save_url=model_save_url,
        **kwargs
    )
