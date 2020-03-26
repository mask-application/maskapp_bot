import logging
import os

from telegram import Update, User
from telegram.error import Unauthorized, BadRequest, TelegramError
from telegram.ext import CommandHandler, CallbackContext, MessageHandler, Filters, run_async
from telegram.ext import Updater

import mask_bandi
from configs import CONFIG_ID, bot_token
from utils import config_logging

please_wait_message = "لطفا منتظر بمانید. (با توجه به زیادی درخواست ها ممکن است پاسخگویی به درخواست شما " \
                      "از چند ثانیه تا چند ساعت به طول انجامد)"
resend_message = ("لطفا برای ماسک نصب کردن برای تصویر پروفایل تان دستور /mask_my_avatar را بزنید و یا ",
                  "لطفا برای ماسک نصب کردن برای تصویر پروفایل تان دستور /mask_my_avatar را بزنید و یا ",
                  "")[CONFIG_ID] + 'برای ماسک نصب کردن روی یک عکس، آن را ارسال کنید.'
start_message = "سلام هم‌وطن.\nاین" \
                " ربات به شما کمک می‌کند که با افزودن ماسک به عکس پروفایل‌تان در رسانه‌های اجتماعی و پیام‌رسان‌های " \
                "موبایلی به پویش مردمی علیه کرونا بپیوندید و\n از هم‌وطنانتان دعوت کنید با نصب اپلیکیشن «ماسک» " \
                "به کاهش قربانیان کرونا در ایران کمک کنند. لطفا برای افزودن ماسک به تصویر پروفایل‌تان روی دستور " \
                "/mask_my_avatar بزنید. برای ماسک نصب کردن روی یک عکس دیگر هم آن عکس را همین‌جا ارسال کنید. به محض " \
                "اماده شدن عکس ماسک‌دارتان برایتان ارسال می‌شود. ببخشید اگر به خاطر ترافیک بالا چند دقیقه‌ای طول کشید." \
                " عکس باید از روبرو گرفته شده باشد. عکس‌تان فقط برای خودتان ارسال می‌شود و جایی نگه داشته نمی‌شود.\n " \
                "https://cafebazaar.ir/app/ir.covidapp.android"

_simple_bot = _updater = _request = None

Configs = [("socks5h://localhost:4567", 'https://api.telegram.org/bot', 'https://api.telegram.org/file/bot'),
           (None, 'https://api.telegram.org/bot', 'https://api.telegram.org/file/bot'),
           (None, 'https://tapi.bale.ai/', 'https://tapi.bale.ai/file/'),
           ]

proxy_url, base_url, base_file_url = Configs[CONFIG_ID]


@run_async
def start(update: Update, _context: CallbackContext):
    chat, user = update.effective_chat, update.effective_user
    logging.debug('/start from user:{%s} chat:{%s}' % (user.id, chat and chat.id))
    update.message.reply_photo(photo=open("example.jpg", "rb"), caption=start_message)


@run_async
def mask_my_avatar(update: Update, _context: CallbackContext):
    chat, user = update.effective_chat, update.effective_user
    logging.debug('/mask from user:{%s} chat:{%s}' % (user.id, chat and chat.id))
    use_my_avatar(update, _context)


def get_photo(photos):
    suitable = [p for p in photos if
                p['width'] < 4000 and p['height'] < 4000 and p['file_size'] < 4000000]
    suitable.sort(key=lambda p: p['file_size'])
    return suitable[-1] if suitable else None


def use_my_avatar(update: Update, context: CallbackContext):
    user = update.effective_user
    assert isinstance(user, User)
    if update.message.photo:
        photos = update.message.photo
    elif update.message.text == "/mask_my_avatar" and CONFIG_ID != 2:
        photos = user.get_profile_photos()
        if photos['total_count'] == 0:
            update.message.reply_text(
                'شما تصویر پروفایلی ندارید. اگر در تنظیمات امنیتی تلگرام مشخص کرده‌اید که فقط '
                'مخاطبین شما دسترسی به تصویر پروفایلتان داشته باشند،'
                ' لطفا این تنظیم را به «همه» تغییر'
                'دهید. ')
            return
        photos = photos['photos'][0]
    else:
        update.message.reply_text(resend_message)
        return
    update.message.reply_text(please_wait_message)
    logging.info("userid:%s photo:%s" % (user.id, photos))
    photo = get_photo(photos)
    if not photo:
        update.message.reply_text('متاسفانه عکس شما دارای اندازه مناسبی نیست' + "\n" + resend_message)
        return
    image = context.bot.get_file(photo['file_id'])
    image_file_name = 'images/%d.jpg' % user.id
    image_file_name_output_m2 = 'images/%d_m2.out.jpg' % user.id
    image_file_name_output_m4 = 'images/%d_m4.out.jpg' % user.id
    image.download(image_file_name)

    try:
        mask_bandi.main(source="file", input_dir=image_file_name, output_dir=image_file_name_output_m2,
                        decorate=True, method=2)
        mask_bandi.main(source="file", input_dir=image_file_name, output_dir=image_file_name_output_m4,
                        decorate=True, method=4)
    except:
        update.message.reply_text("امکان ماسک نصب کردن برای تصویر پروفایل شما وجود ندارد." + "\n" + resend_message)
        return

    f_m2 = open(image_file_name_output_m2, "rb")
    f_m4 = open(image_file_name_output_m4, "rb")
    update.message.reply_photo(photo=f_m2,
                               caption='برای تغییر عکس پروفایل‌تان به این عکس باید '
                                       'این تصویر را روی گوشی‌تان ذخیره کنید و سپس در بخش تنظیمات پروفایل‌تان'
                                       ' در تلگرام/اینستاگرام/توییتر آن را تصویر جدید پروفایل‌تان کنید.' +
                                       "\n" + resend_message)
    update.message.reply_photo(photo=f_m4,
                               caption='برای تغییر عکس پروفایل‌تان به این عکس باید '
                                       'این تصویر را روی گوشی‌تان ذخیره کنید و سپس در بخش تنظیمات پروفایل‌تان'
                                       ' در تلگرام/اینستاگرام/توییتر آن را تصویر جدید پروفایل‌تان کنید.' +
                                       "\n" + resend_message)
    f_m2.close()
    f_m4.close()
    try:
        os.remove(image_file_name)
        os.remove(image_file_name_output_m2)
        os.remove(image_file_name_output_m4)
    except Exception:
        pass


def get_updater():
    global _updater, _request
    if _updater is None:
        _updater = Updater(bot_token, use_context=True, request_kwargs={'proxy_url': proxy_url}, base_url=base_url,
                           base_file_url=base_file_url)

    return _updater


def error_callback(_bot, _update, error):
    try:
        raise error
    except (Unauthorized, BadRequest, TelegramError):
        logging.exception('Telegram error', extra={'update': _update})
    except:
        logging.warning('Telegram warning', extra={'update': _update})


def main():
    updater = get_updater()
    dispatcher = updater.dispatcher
    dispatcher.add_error_handler(error_callback)
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('mask_my_avatar', mask_my_avatar))
    dispatcher.add_handler(MessageHandler(Filters.all, mask_my_avatar))
    updater.start_polling(timeout=10)
    updater.idle()


if __name__ == '__main__':
    config_logging(logging.INFO, "telegram_bot.log")
    os.environ["FACEALIGNMENT_USERDIR"] = os.path.abspath("images/")
    main()
