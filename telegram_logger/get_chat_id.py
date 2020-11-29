import logging
from argparse import ArgumentParser, Namespace

from telegram import Update
from telegram.ext import CallbackContext, CommandHandler, Filters, \
    MessageHandler, \
    Updater

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def parser_args():
    parser = ArgumentParser()
    parser.add_argument('--token', type=str, required=True,
                        help='telegram token api')
    return parser.parse_args()


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! Add me to a channel to which I will send '
                              'messages. Also forward me a message from that '
                              'group so I can get group id.')


def recognize_chat_id(update: Update, context: CallbackContext) -> None:
    logger.info(update)
    chat_id = update.message.forward_from_chat.id
    logger.info('Your Chat id is "{0}"'.format(chat_id))
    update.message.reply_markdown('Chat id is `{0}`'.format(chat_id))
    context.bot.send_message(chat_id, 'LoggerBot is configured!')


def main(args: Namespace):
    updater = Updater(args.token, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.forwarded, recognize_chat_id))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main(parser_args())
