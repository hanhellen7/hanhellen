import tushare as ts
import datetime
import pandas as pd
from data_util import read_csv, exist_file
ts.set_token('d4043584d7a5733e37f06898354e7ccfdc46f0ee26c7f0d2b1a11f21')
pro = ts.pro_api()
"""
交易策略
读取估值列表
获取估值信息
如果在低谷区间,提示买入

"""
class StockModel:
    def __init__(self, root_dir):
        self.token = 'd4043584d7a5733e37f06898354e7ccfdc46f0ee26c7f0d2b1a11f21'
        self.code_path = root_dir + "stock_code.csv"
        self.mv_path = root_dir + "mv.csv"
        self.mv_code_path = root_dir + "mv_code.csv"
        self.mv_res_path = root_dir + "mv_res_"+datetime.datetime.now().strftime('%Y-%m-%d')
        self.load_ts()
        self.download_code()
        self.load_mv_with_code()

    def load_ts(self):
        ts.set_token(self.token)
        self.ts = ts.pro_api()

    def junlin_data(self):
        pass

    def add_mv(self):
        csv_res = read_csv(self.mv_path)

    def download_code(self):
        # 得到股票代码
        if exist_file(self.code_path):
            print(f"exists code file {self.code_path}")
        else:
            stock_code = self.ts.stock_basic(exchange='',
                                list_status='L',
                                fields='ts_code,symbol,name')
            stock_code.to_csv(self.code_path,
                              encoding="utf-8",
                              mode="w+",
                              header=True, index=False)
        self.stock_code = pd.read_csv(self.code_path)

    def load_mv_with_code(self):
        if exist_file(self.mv_code_path):
            self.mv_df = read_csv(self.mv_code_path)
            print(f"exists code file {self.mv_code_path}")
        else:
            self.add_code2mv()
            self.mv_df.to_csv(self.mv_code_path,
                              encoding="utf-8",
                              mode="w+",
                              header=True, index=False)

    def add_code2mv(self):
        mv_df = read_csv(self.mv_path)
        code_lst = []
        for name in mv_df['stock_name']:
            s_code = self.stock_code[self.stock_code.name==name]['ts_code']
            s_code = list(s_code)
            if s_code:
                s_code = s_code[0]
            else:
                s_code = "unknown"
            code_lst.append(s_code)
        mv_df['code'] = code_lst

        mv_df.to_csv(self.mv_code_path,
                     encoding="utf-8",
                     mode="w+",
                     header=True, index=False)
        self.mv_df = mv_df

    def get_junlin_data(self):
        if exist_file(self.mv_res_path):
            self.res_df = read_csv(self.mv_res_path)
        else:
            mv_df = self.mv_df[self.mv_df.code!='unknown'].copy()
            status_lst = []
            pe_lst = []
            pb_lst = []
            rate_lst = []
            cur_lst = []
            for code in mv_df.iterrows():
                print(code)
                low = code[1]["low"]
                high = code[1]["high"]
                middle = code[1]["middle"]
                code = code[1]['code']
                cur_value, pe, pb = self.get_mv(code)
                if cur_value >= high:
                    status = "high"
                    rate = (cur_value - high) / middle
                elif cur_value <= low:
                    status = "low"
                    rate = (cur_value-low) / middle
                else:
                    status = "middle"
                    rate = (cur_value-middle)/middle
                cur_lst.append(cur_value)
                rate_lst.append(rate)
                status_lst.append(status)
                pe_lst.append(pe)
                pb_lst.append(pb)
            mv_df['rate'] = rate_lst
            mv_df['cur'] = cur_lst
            mv_df['status']  = status_lst
            mv_df['pe']  = pe_lst
            mv_df['pb']  = pb_lst
            mv_df.to_csv(self.mv_res_path,
                              encoding="utf-8",
                              mode="w+",
                              header=True, index=False)
            self.res_df = mv_df
        return self.res_df

    def pack_reply(self, df):
        reply = ""
        status_trans_dct = {
                        "low":"低",
                        "high":"高",
                        "middle":"中"
                        }
        for d in df.iterrows():
            low = d[1]["low"]
            high = d[1]["high"]
            middle = d[1]["middle"]
            status = d[1]['status']
            name = d[1]['stock_name']
            rate = d[1]['rate']
            cur = d[1]['cur']
            t_status = status_trans_dct.get(status)
            reply += f"{name},{cur:.0f},{low}~{middle}~{high},{rate:.0%}\n"
        if reply == "":
            reply = "当前暂未发现该类型股票"
        return reply


    def junlin_low(self, type='low'):
        mv_df = self.get_junlin_data()
        low_df = mv_df[mv_df.status==type]
        print(low_df)
        reply = self.pack_reply(low_df)
        return reply

    def junlin_middle_low(self, type='middle'):
        mv_df = self.get_junlin_data()
        low_df = mv_df[(mv_df.status==type)&(mv_df.rate<0)]
        reply = self.pack_reply(low_df)
        return reply

    def junlin_middle_high(self, type='middle'):
        mv_df = self.get_junlin_data()
        low_df = mv_df[(mv_df.status==type)&(mv_df.rate>0)]
        reply = self.pack_reply(low_df)
        return reply

    def junlin_high(self, type='high'):
        mv_df = self.get_junlin_data()
        df = mv_df[mv_df.status==type]
        reply = self.pack_reply(df)
        return reply


    def get_mv(self, code):
         """
         得到实时市值
         """
         res = pro.daily_basic(ts_code=code,
                         # trade_date='2020726',
                         fields='ts_code,trade_date,turnover_rate,total_mv,pe,pb')
         res = pd.DataFrame(data=res)
         return res.loc[0,'total_mv']/10000, res.loc[0,'pe'],res.loc[0,'pb']

if __name__ == '__main__':
    print(ts.__version__)
    # my_chice()
    # get_account_num()

    s = StockModel("../data/")
    s.download_code()
    s.load_mv_with_code()
    s.junlin_high()
