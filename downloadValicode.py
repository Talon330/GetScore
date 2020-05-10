import requests
import os
from PIL import Image
import random
import time

if not os.path.exists("./image/"):
    os.makedirs("./inage/")
IMAGE_URL = "https://yz.scu.edu.cn/User/Valicdoe"

getnum = 200 #验证码周期次数
def request_download(i,j):
    """获取验证码并保存,命名方式是i_j.jpg，默认保存到当前目录下的image文件夹"""
    r = requests.get(IMAGE_URL)
    with open('./image/{}{}.jpg'.format(i,j), 'wb') as f:
        f.write(r.content)

for i in range(getnum):
    for j in range(3): #每采集三次休息几秒
        try:
            request_download(i,j)
            print('download img-{}{}'.format(i,j))
        except:
            print('download img-{}{} error!'.format(i,j))
    time.sleep(random.random()*5)
    print("\nstart again")