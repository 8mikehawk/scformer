import time


class Timer():
    def __init__(self):
        self.last_time = time.time()
        self.remain_hour = None
        self.remain_min = None
        self.remain_second = None

    def get_remain_time(self, idx, max_epoch):
        remain_time=(time.time()-self.last_time)*(max_epoch-idx-1)
        self.last_time=time.time()

        self.remain_hour = int(remain_time / 3600)
        self.remain_min = int((remain_time - self.remain_hour * 3600) / 60)
        self.remain_second = int(remain_time - (3600 * self.remain_hour) - (60 * self.remain_min))

        # print(f"hour {self.remain_hour}, min {self.remain_min}, second {self.remain_second}")

        return {"hour": self.remain_hour, "min": self.remain_min, "second": self.remain_second}