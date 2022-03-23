from datetime import datetime
import time


while True:
    with open('readme.txt', 'a') as f:
        time_now = datetime.now().strftime("%H:%M:%S")
        # print(time_now)
        f.writelines(time_now)
        f.write('\n')
        time.sleep(3)
        print("time log")