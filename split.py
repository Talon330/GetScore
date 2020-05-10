from PIL import Image
import numpy as np
import queue
import os
import time

tobesplit="./image/" #待分割训练集图片存放路径
denoised="./denoised/" #去噪后训练集图片存放路径
spliterror="./denoserr/" #分割出错的训练集图片存放路径
splitdone="./singlechar/" #分割后训练集图片存放路径
binarythresh = 80 #二值化时简单去除噪声阈值
denoiethresh = 180 #连通域去除周围点阈值
image_suffix = "jpg" #分割后保存的格式


def depoint(img, denoiethresh):
    """传入二值化后的图片进行降噪"""
    pixdata = np.array(img)
    w,h = img.size
    for x in range(h):
        for y in range(w):
            count = 0
            if x==0 or x==h-1 or y==0 or y==w-1:
                pixdata[x,y]=255
            else:
                if pixdata[x,y-1] > denoiethresh:
                    count = count + 1
                if pixdata[x,y+1] > denoiethresh:
                    count = count + 1
                if pixdata[x-1,y] > denoiethresh:
                    count = count + 1
                if pixdata[x+1,y] > denoiethresh:
                    count = count + 1
                if pixdata[x-1,y-1] > denoiethresh:
                    count = count + 1
                if pixdata[x-1,y+1] > denoiethresh:
                    count = count + 1
                if pixdata[x+1,y-1] > denoiethresh:
                    count = count + 1
                if pixdata[x+1,y+1] > denoiethresh:
                    count = count + 1
                if count > 4:
                    pixdata[x,y] = 255
    return Image.fromarray(pixdata)


def binar(image,thresh): 
    """尽可能地过滤绿色像素并二值化"""
    image=image.convert('RGB')
    w,h = image.size
    thresh=thresh
    im=np.array(image)
    for i in range(w):
        for j in range(h):
            r, g, b = image.getpixel((i, j))
            value = 0.3*r + 0.2*g + 0.5*b #验证码的字符是蓝色和红色，可以把绿色权重设置小一点
            if value < thresh:
                image.putpixel((i, j), (0, 0, 0))
            else:
                image.putpixel((i, j), (255, 255, 255))
    return Image.fromarray(im).convert("1")


def fillup(image): 
    """若一个点四周超过5个点都是黑色，则该点也变为黑色"""
    offset = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    w,h = image.size
    im = np.array(image)
    for x in range(1,h-1):
        for y in range(1,w-1):
            count=0
            around=0
            for x_offset,y_offset in offset:
                x_c, y_c = x + x_offset, y + y_offset
                if im[x_c, y_c]==0:
                    count+=1
                if x_offset==0 or y_offset==0 and im[x_c, y_c]==0:
                    around+=1
            if count > 5 and im[x,y] == 1 or around >= 4:
                    im[x,y] = 0
    return Image.fromarray(im)




def cfs(img):
    """
    # 传入二值化后的图片进行连通域分割 CFS连通域分割法
    """
    pixdata = img.load()
    w,h = img.size
    visited = set()
    q = queue.Queue()
    offset = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    cuts = []
    for x in range(w):
        for y in range(h):
            x_axis = []
            if pixdata[x,y] == 0 and (x,y) not in visited:
                q.put((x,y))
                visited.add((x,y))
            while not q.empty():
                x_p,y_p = q.get()
                for x_offset,y_offset in offset:
                    x_c,y_c = x_p+x_offset,y_p+y_offset
                    if (x_c,y_c) in visited:
                        continue
                    visited.add((x_c,y_c))
                    try:
                        if pixdata[x_c,y_c] == 0:
                            q.put((x_c,y_c))
                            x_axis.append(x_c)
                    except:
                        pass
            if x_axis:
                min_x,max_x = min(x_axis),max(x_axis)
                if max_x - min_x >  3 and max_x - min_x < 18:
                    # 宽度小于3的认为是噪点，介于3和18之间大概率包含了两个字符
                    cuts.append((min_x,max_x))
                elif max_x - min_x >= 18:
                    mid=(min_x + max_x)//2
                    cuts.append((min_x,mid))
                    cuts.append((mid,max_x))


    return cuts


def split():
    filelist = os.listdir(tobesplit)
    for fn in filelist:
        path = os.path.join(tobesplit, fn)
        image = Image.open(path)
        d_img = depoint(image, denoiethresh)
        c_img = binar(d_img, binarythresh)
        b_img = fillup(c_img)
        if not os.path.exists(denoised):
            os.makedirs(denoised)
        p1=os.path.join(denoised, fn)
        b_img.save(p1)
        v = cfs(b_img)
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
            print("分割位置{}有错误\n".format(v))
            if not os.path.exists(spliterror):
                os.makedirs(spliterror)
            p2=os.path.join(spliterror, fn)
            b_img.save(p2)
        else:
            if cuts[0,0]>0 and cuts[3,3]<57:
                for i,n in enumerate(cuts):
                    if not os.path.exists(splitdone):
                        os.makedirs(splitdone)
                    temp = b_img.crop(n) # 调用crop函数进行切割
                    timec = str(time.time()).replace(".", "")
                    p = os.path.join(splitdone, "{}_{}.{}".format(fn[i], timec, image_suffix))
                    temp.resize((24,44)).save(p)
            else:
                print("分割位置{}超出边界\n\n".format(cuts))



if __name__ == '__main__':
    split()