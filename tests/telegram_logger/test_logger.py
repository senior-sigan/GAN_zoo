import os

import pytest
import torch

from gans_zoo.utils import tensor_to_file_like_object
from telegram_logger.logger import TelegramLogger


def setup_tg():
    token = os.getenv('TG_TOKEN')
    chat_id = os.getenv('TG_CHAT_ID')
    if token is None:
        pytest.skip('Test requires TG_TOKEN env variable to be set')
    if chat_id is None:
        pytest.skip('Test requires TG_CHAT_ID env variable to be set')

    return TelegramLogger(
        token=token,
        chat_id=int(chat_id),
        module_name=__name__,
    )


def test_send_text():
    tg_logger = setup_tg()
    tg_logger.write_text('Test message')


def test_send_image_as_tensor():
    tg_logger = setup_tg()
    tensor = torch.rand(size=(3, 256, 256))
    file = tensor_to_file_like_object(tensor)
    tg_logger.write_image(file)
