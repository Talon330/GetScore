import split
import requests
import os, time
import io, random
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import numpy as np
from cnnlib.recognition_object import Recognizer

nlpath = "复试名单.csv"
filedpath = "查询失败的信息.csv"

col1 = "姓名"
col2 = "考生编号"
col3 = "报考专业名称"
score = dict(col4 = "政治理论",
    col5 = "外国语",
    col6 = "业务课一",  #一般来说“业务课一”是数学，“业务课二”是专业课
    col7 = "业务课二",
    col8 = "总分")

image_height = 44 #字符分割后resize的高
image_width = 24
max_captcha = 1 #每次识别的字符数量
char_set = "024678BDFHJLNPRTVXZ" #所有能够识别的字符，根据下载的验证码数据，剔除了从没有出现的字符
model_save_dir = "./model/" #模型保存的路径
captchapath = "./GET_CAPTCHA/" #获取的验证码识别后保存的路径，可以作为验证码训练数据集
image_suffix = "jpg"  #保存的扩展名

namelist = pd.read_csv(nlpath, encoding="utf-8")
rownum = namelist.shape[0]
for col in score.values():
    namelist[col]=""

s = requests.Session()

get_url = 'https://yz.scu.edu.cn/score'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
}
r = s.get(url=get_url, headers=headers)
R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)


def captcha_recognition(times):
    global vcd, vcd1, vcd2
    captcha_url = 'https://yz.scu.edu.cn/User/Valicdoe'
    r = s.get(captcha_url)
    # with open('./yanzhengma.jpg', 'wb') as fp:
    #     fp.write(r.content)
    # image = Image.open("./yanzhengma.jpg")#('./yanzhengma.jpg')
    image = Image.open(io.BytesIO(r.content))
    w = image.size[0]
    de_img = split.depoint(image, split.denoiethresh)
    bi_img = split.binar(de_img, split.binarythresh)
    fi_img = split.fillup(bi_img)
    v = split.cfs(fi_img)
    print('分割位置：', v)
    try:    
        v=np.array(v)
        cuts=np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if j==0:
                    if v[i,0]-1 < 0:
                        cuts[i,j]=v[i,0]
                    else:
                        cuts[i,j]=v[i,0]-1
                elif j==1:
                    cuts[i,j]=0
                elif j==2:
                    cuts[i,j]=v[i,1]+1
                else:
                    cuts[i,j]=22
    except:
        print("验证码字符分割位置计算错误，1S后重新获取验证码\n\n")
        time.sleep(1)
        if times > 0:
            times -= 1
            captcha_recognition(times)
        else:
            return 0
    else:
        #cuts = [(4,0,16,22),(16,0,27,22),(27,0,38,22),(38,0,53,22)] #'./User/0J0N.jpg'的较好分割位置
        vcd1 = []
        if cuts[0,0]>0 and cuts[3,3]<w-1: #若没有超出图片边界
            for i,n in enumerate(cuts):
                temp = fi_img.crop(n) # 调用crop函数进行切割
                temp = temp.resize((image_width, image_height))
                
                temp.save("temporary.jpg") #不转存识别效果就很差，原因不明，可能由于Recognizer对象里需要灰度化，本图像已经完成二值化
                temp = Image.open("temporary.jpg")
                
                start = time.time()
                key = R.rec_image(temp)
                end = time.time()
                vcd1.append(key)
        else:
            print("分割位置超出边界，分割失败，1S后重新获取验证码\n\n")
            time.sleep(1)
            if times > 5:
                times -= 1
                captcha_recognition(times)
            else:
                return 0
        vcd2 = [str(i) for i in vcd1]
        vcd = ''.join(vcd2)
        print("识别结果：{}".format(vcd))
        if not os.path.exists(captchapath):
            os.makedirs(captchapath)
        path = os.path.join(captchapath, "{}_{}.{}".format(vcd, str(start).replace(".", ""), image_suffix))
        print("识别用时{}\n".format(end - start))
        print("验证码图片将保存到{}".format(path))
        image.convert('RGB').save(path)
    return vcd


def cleandata(i, txt): #
    """数据清洗，只保留成绩，i用来标定行数，txt传入网页数据"""
    global namelist
    soup = BeautifulSoup(txt, 'html.parser')
    for idx, tr in enumerate(soup.find_all('tr')):
        spans = tr.find_all('span')
        namelist.loc[i,score["col{}".format(idx+4)]]=spans[0].contents[0]
        namelist.to_csv(nlpath, encoding="utf-8", index=False)
    print("编号{}同学的成绩已保存\n".format(i))



failed = pd.DataFrame(columns=[col1, col2, col3])
for i in range(rownum):
    for j in range(3): #每人查询最多允许失败三次
        times = 5 #设定每次验证码识别失误阈值
        vcode = captcha_recognition(times)
        if vcode != 0:
            post_url = 'https://yz.scu.edu.cn/score/Query/--'
            data = {'zjhm': "",
                    'xm': namelist.loc[i,col1],
                    'vcode': vcode,
                    'ksbh': namelist.loc[i,col2]}
            r = s.post(post_url, headers=headers, data=data)
            if "校验码错误或失效" not in r.text and namelist.loc[i,col1] in r.text: #验证码识别错误返回信息会包含“校验码错误或失效！”,正确识别应该包含待查询人的名字
                cleandata(i, r.text)
                break
        elif j==2 or vcode==0: #验证码获取次数和识别次数均超过限制
            failed = failed.append(namelist.loc[i], ignore_index=True)
            print("编号{}同学的成绩查询失败，将保存该同学信息到{}，建议手动查询\n\n".format(i, filedpath))
            failed.to_csv(filedpath,encoding="utf-8", index=False)
    print("查询完一人，休息几秒\n")
    time.sleep(random.random()*5) 
if os.path.exists("temporary.jpg"):
    os.remove("temporary.jpg")   

