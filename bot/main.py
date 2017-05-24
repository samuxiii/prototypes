import dtree as dt
import telebot
from telebot import types
import time

TOKEN = ''

bot = telebot.TeleBot(TOKEN)
smart = 0;

def listener(messages):

    for m in messages:
        if m.content_type == 'text':
            cid = m.chat.id
            print "[" + str(cid) + "]: " + m.text

bot.set_update_listener(listener)

#global variables
smart = 0
height = 0
weight = 0
foot = 0

# func #
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "How are you doing?\nAvailable commands:\n/photo\n/say\n/gender")

@bot.message_handler(commands=['photo'])
def command_photo(m):
    cid = m.chat.id
    bot.send_photo( cid, open( 'l.jpg', 'rb'))

@bot.message_handler(commands=['say'])
def command_say(m):
    cid = m.chat.id
    bot.send_message( cid, 'fuck fuck.. XIII')

@bot.message_handler(func=lambda message: smart > 0)
def smart_actions(m):
    global smart, weight, height, foot
    cid = m.chat.id
    if (smart == 1):
        height = m.text
        markup = types.ForceReply(selective=False)
        bot.send_message( cid, 'How much do you weigh?', reply_markup=markup)
        smart = 2
    elif (smart == 2):
        weight = m.text
        markup = types.ForceReply(selective=False)
        bot.send_message( cid, 'What size shoes do you wear?', reply_markup=markup)
        smart = 3
    else:
        foot = m.text
        smart = 0
        prediction = dt.predictGender(height, weight, foot)
        result_predicted = "According to your answer, you should be a " + str(prediction + "...")
        bot.send_message(cid, result_predicted)

@bot.message_handler(commands=['gender'])
def command_smart(m):
    global smart, height
    cid = m.chat.id
    smart = 1;
    markup = types.ForceReply(selective=False)
    bot.send_message( cid, 'How tall are you?', reply_markup=markup)
# end func #
bot.polling(none_stop=True)
