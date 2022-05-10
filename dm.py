import re
from stock import StockModel
import stock
import random
import yaml
class Dm:
    def __init__(self, root_dir):
        self.policy_path = root_dir + "policy.yml"
        self.load_stock()
        self.load_policy()

    def load_stock(self):
        self.stock =  StockModel("../data/")

    def load_policy(self):
        with open(self.policy_path, 'r', encoding='utf-8') as f:
            self.policy = yaml.load(f)
            print(self.policy)

    def update_info(self):
        # 自动更新君临数据
        self.stock.get_junlin_data()

        pass

    def parse_policy(self, msg):
        for policy_key, policy_value in self.policy.items():
            if msg == '君临高估':
                return self.stock.junlin_high()
            elif msg == '君临低估':
                return self.stock.junlin_low()
            elif msg == '君临偏低':
                return self.stock.junlin_middle_low()
            elif msg == '君临偏高':
                return self.stock.junlin_middle_high()

            else:
                return self.random_reply(msg)
            # if re.search(policy_key, msg):
            #     [policy_value](msg)

    def random_reply(self, msg):
        rule1 = re.compile("[吗啊]$")
        if msg == "你好":
            return random.choice(["我很好","哈哈哈","你也好","同好"])
        elif msg == "你是谁":
            return random.choice(["不告诉你","我是你爸爸爸","你猜","我是小可爱"])
        elif re.search(rule1, msg):
            re_msg = re.sub(rule1,"",msg)
        else:
            re_msg = random.choice(["看杂技吗","听歌吗","看电视吗","我给你讲个笑话","我给你表演个节目"])

        if msg in ["好","行","可以","好啊"]:
            return "我不会"
        return re_msg


    def run_activate(self, func):
        pass

if __name__ == "__main__":
    dm = Dm("../data/")
    dm.load_policy()
    print(dm.parse_policy('君临高估'))
    print(dm.parse_policy('君临低估'))
    print(dm.parse_policy('君临偏低'))
    print(dm.parse_policy('君临偏高'))
