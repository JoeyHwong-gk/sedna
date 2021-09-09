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

from abc import ABC, abstractmethod
from sedna.common.config import BaseConfig
from sedna.common.class_factory import ClassFactory, ClassType


class AbstractTransmitter(ABC):
    """
    Abstract class of Transmitter, which provides base transmission
    interfaces between edge and cloud.
    """

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def send(self, data):
        pass


@ClassFactory.register(ClassType.TRAM, alias="ws")
class WSTransmitter(AbstractTransmitter, ABC):
    """
    An implementation of Transmitter based on WebSocket.
    """

    def __init__(self, config=None):
        self.parameters = {}

    def recv(self):
        pass

    def send(self, data):
        pass


@ClassFactory.register(ClassType.TRAM, alias="s3")
class S3Transmitter(AbstractTransmitter, ABC):
    """
    An implementation of Transmitter based on S3 protocol.
    """

    def __init__(self, config=None):
        if not config:
            config = BaseConfig
        self.parameters = {
            "s3_endpoint_url": config.s3_endpoint_url,
            "s3_bucket": config.agg_data_path,
            "access_key": config.access_key_id,
            "secret_key": config.secret_access_key,
        }

    def recv(self):
        pass

    def send(self, data):
        pass
