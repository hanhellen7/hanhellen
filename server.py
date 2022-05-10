import hashlib
import re
import random
from dm import Dm
import time
from flask import Flask, request, make_response
import xml.etree.ElementTree as ET
from wechatpy import parse_message
from wechatpy.replies import TextReply
WX_TOKEN = 'fancy'
app = Flask(__name__)

@app.route('/')
def index():
        return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello World'
@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id
@app.route('/wechat_api/',methods=['GET','POST'])
def wechat():
    print(request.data)
    if request.method == 'GET':
        token = WX_TOKEN
        data = request.args
        signature = data.get('signature', '')
        timestamp = data.get('timestamp', '')
        nonce = data.get('nonce', '')
        echostr = data.get('echostr', '')
        s = sorted([timestamp, nonce, token])
        # 字典排序
        s = ''.join(s)
        if hashlib.sha1(s.encode('utf-8')).hexdigest() == signature:
        # 判断请求来源，并对接受的请求转换为utf-8后进行sha1加密
            print(echostr)
            response = make_response(echostr)
            return response
    else:
        xml = request.data
        msg = parse_message(xml)
        print(dir(msg))
        print(msg.source)
        print(msg.target)
        if msg.type == 'text':
            dm = Dm("../data/")
            content =  dm.parse_policy(msg.content)
            try:
                reply = TextReply(content=content,message=msg)
                r_xml = reply.render()
                # 获取唯一标记用户的openid，下文介绍获取用户信息会用到
                openid = msg.source
                return make_response(r_xml)
            except Exception as e:
                print(e)
                pass

def reply_policy(msg):
    rule1 = re.compile("[吗啊]$")
    if re.search(rule1, msg):
        re_msg = re.sub(rule1,"",msg)
    else:
        re_msg = random.choice(["看杂技吗","听歌吗","看电视吗","我给你讲个笑话","我给你表演个节目"])

    if msg in ["好","行","可以","好啊"]:
        return "我不会"
    return re_msg

if __name__ == '__main__':
    # 设置守护进程
    app.run(port=8080)
