# 记录
class Grid:

    def __init__(self):
        pass

    def record_add(self, code, num, price):
        """
          记录加仓点
        """
        pass

    def record_buy(self, code, num, price):
        """
          记录减仓点
        """
        pass

    def evaluate_value(self, code, num, price, period):
        """
          计算估值区间
        """
        pass
class DataProcess:

    def __init__(self):
        pass

    def get_all_value(self, code, period):
        """
          得到period的所有估值
        """
