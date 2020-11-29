from telegram.ext import Updater


class TelegramLogger(object):
    def __init__(
        self,
        token: str,
        chat_id: int,
        module_name: str = '',
        request_kwargs=None,
    ):
        self.bot = Updater(token, request_kwargs=request_kwargs).bot
        self.chat_id = chat_id
        self.module_name = module_name

    def write_text(self, text):
        """
        Send a text message to the chat.

        :param text:
        :return:
        """
        self.bot.send_message(self.chat_id, self.module_name + "\n" + text)

    def write_image(self, file_obj, caption=None):
        """
        Send an image to the chat.

        :param file_obj: file like object, probably from open('img.jpg', 'rb')
        :param caption:
        :return:
        """
        self.bot.send_photo(self.chat_id, file_obj, caption=caption)
