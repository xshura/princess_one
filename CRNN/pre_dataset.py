# _*_ coding:utf-8 _*_
"""
模型类
CNN + LSTM + CTC
create_time: 2018/4/15
creator ：shura
"""
import re
import cv2
import numpy as np
import os


# 调试方法
def demo_test():
    dir = "T1mLyYXvRaXXXXXXXX_!!1-item_pic.gif.jpg"
    img = cv2.imread(dir)
    f = open("T1mLyYXvRaXXXXXXXX_!!1-item_pic.gif.txt", encoding='UTF-8')
    regions = f.read()
    region_line = regions.split()
    # for sites in region_line:
    sites = region_line[0].split(',')
    #     sites = sites.split(',')
    x1 = np.float32(sites[0])
    y1 = np.float32(sites[1])
    x2 = np.float32(sites[2])
    y2 = np.float32(sites[3])
    x3 = np.float32(sites[4])
    y3 = np.float32(sites[5])
    x4 = np.float32(sites[6])
    y4 = np.float32(sites[7])
    # 划线框住目标区域
    A, B, C, D = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    cv2.line(img, A, B, (255, 0, 255), 2)
    cv2.line(img, B, C, (255, 0, 255), 2)
    cv2.line(img, C, D, (255, 0, 255), 2)
    cv2.line(img, D, A, (255, 0, 255), 2)
    # 缩放目标点坐标 左上 右上 右下 左下
    src_points = np.float32([[x1, y1], [x4, y4], [x3, y3], [x2, y2]])
    # 计算矫正后目标矩形 width 和 high
    highA = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    highB = np.sqrt(((x3 - x4) ** 2) + ((y3 - y4) ** 2))
    maxHeight = max(int(highA), int(highB))
    widthA = np.sqrt(((x4 - x1) ** 2) + ((y1 - y4) ** 2))
    widthB = np.sqrt(((x2 - x3) ** 2) + ((y3 - y2) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    if maxHeight < 1:  maxHeight = 1
    if maxWidth < 1:  maxWidth = 1
    # 矫正后输出坐标设置
    dst_points = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])
    # 进行仿射变换
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    target = cv2.warpPerspective(img, projective_matrix, (0, 0))
    target = target[0:maxHeight, 0:maxWidth]
    cv2.imshow("image", img)      # 显示图片
    cv2.imshow("target", target)  # 显示图片
    # 另存为图像
    # save_dir = "D:\\work\\tianchi\\pre-treatment\\train_set\\" + sites[8] + ".jpg"
    cv2.imwrite(save_dir, target)
    cv2.waitKey(0)


# 图片预处理方法
def pre_image(img_dir, txt_dir, save_dir):
    num = 0
    labelfile = open('labels.txt', 'a')
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            filepath = os.path.join(root, file)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in ["jpg","png"]:  # 遍历找到指定后缀的文件名["jpg",png]等
                img = cv2.imread(filepath)
                txt_path = txt_dir + os.path.splitext(file)[0] + ".txt"
                f = open(txt_path, encoding='UTF-8')
                regions = f.readlines()
                for sites in regions:
                    try:
                        sites = sites.strip('\n').split(',')
                        x1 = np.float32(sites[0])
                        y1 = np.float32(sites[1])
                        x2 = np.float32(sites[2])
                        y2 = np.float32(sites[3])
                        x3 = np.float32(sites[4])
                        y3 = np.float32(sites[5])
                        x4 = np.float32(sites[6])
                        y4 = np.float32(sites[7])
                        # 若是###则丢弃
                        if sites[8] == "###": continue
                        # 缩放目标点坐标 左上 右上 右下 左下
                        src_points = np.float32([[x1, y1], [x4, y4], [x3, y3], [x2, y2]])
                        # 计算矫正后目标矩形 width 和 high
                        highA = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
                        highB = np.sqrt(((x3 - x4) ** 2) + ((y3 - y4) ** 2))
                        maxHeight = max(int(highA), int(highB))
                        widthA = np.sqrt(((x4 - x1) ** 2) + ((y1 - y4) ** 2))
                        widthB = np.sqrt(((x2 - x3) ** 2) + ((y3 - y2) ** 2))
                        maxWidth = max(int(widthA), int(widthB))
                        # 因为opencv仿射变换只能接受integer整形，当出现小于1得数需要做处理否则会报错
                        if maxHeight < 1:  maxHeight = 1
                        if maxWidth < 1:  maxWidth = 1
                        # 矫正后输出坐标设置
                        dst_points = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])
                        # 进行仿射变换
                        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                        target = cv2.warpPerspective(img, projective_matrix, (0, 0))
                        target = target[0:maxHeight, 0:maxWidth]
                        # cv2.imshow("image", img)  # 显示图片
                        # cv2.imshow("target", target)  # 显示图片
                        # 另存为图像
                        num += 1
                        file_name = padding_zero(num)
                        save_res = save_dir
                        save_path = save_res + file_name + ".jpg"
                        # 将label编号对应到text文件里面
                        value = file_name+' '+sites[8]+'\n'
                        value.encode("utf-8")
                        labelfile.write(value)
                        if target is None:
                            continue
                        if not os.path.exists(save_res):
                            os.makedirs(save_res)
                        # 若是当前文件报错则丢弃当前文件进入下一次分割输出
                        cv2.imencode(".jpg", target)[1].tofile(save_path)
                    except Exception:
                        print(filepath+"下的"+sites[8]+"Exception异常！！")
                        continue
                    except FileNotFoundError:
                        print(filepath+"下的"+sites[8]+"出现文件找不到异常！！")
                        continue
    labelfile.close()


def padding_zero(number):
    x = 10000000
    if number >= x: return
    res = x + number
    res = str(res)
    return res[1:]




# 输出图片替换其中的特殊字符为对应的ascii值
def validateName(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, lambda m: '_sign_'+str(ord(m.group(0))), title)  # 替换为下划线+ascii值
    return new_title


if __name__ == '__main__':
    img_dir = "D:\\work\\tianchi\\dataset\\image6001-9000\\"
    # img_dir = "D:\\work\\tianchi\\test\\"
    txt_dir = "D:\\work\\tianchi\\dataset\\txt6001-9000\\"
    save_dir = "D:\\work\\tianchi\\train_set\\"
    pre_image(img_dir, txt_dir, save_dir)
    # print(validateName("renhonghuai????"))
    # demo_test()
    # print(padding_zero(23))
