# coding: utf-8
# 第一行的作用是用来声明文本的字符集格式，可以识别和输出中文；

'''
    Python快速入门
'''
import os  # 导入模块os.py
import io
import matplotlib  # 调用函数时候使用matplotlib.函数名()
import numpy as np  # 导入numpy.py，并且用np这个名字来代替numpy
import matplotlib.image as mpimg  # 调用函数: mpimg.函数名()
import matplotlib.pyplot as plt
from scipy.ndimage import filters  # 在scipy.ndimage总仅载入filters模块
from IPython.display import Image  # 函数调用：display.子功能()


# 定义一个函数，名为main，不带输入参数
def main():
    # 调用函数
    simpleCal()
    # List
    # 列表类型
    la, lb = [1, 2, 3], ["苹果", "banana"]
    # python赋值非常灵活，不需要中间变量
    la, lb = lb, la
    # 增强型赋值方法：list comprehension
    new_lb = [i + 2 for i in lb]
    # 用lamda函数来定义一个新的add函数
    add = lambda x, y: x + y
    lc = add(la, lb)
    print("测试add函数", lc)  # 测试add函数 ['苹果', 'banana', 1, 2, 3]
    ld = lc
    lf = []
    lf.extend(lc)
    str_tmp1 = "fruit"
    # 等价于lc[len(lc):]=str_tmp1把字符串打散添加到list末尾
    lc.extend(str_tmp1)
    # 把字符串作为整体添加到最后
    lc.append(str_tmp1)
    for i in lc:
        print("show me the list:" + str(i))
        '''
            show me the list:苹果
            show me the list:banana
            show me the list:1
            show me the list:2
            show me the list:3
            show me the list:f
            show me the list:r
            show me the list:u
            show me the list:i
            show me the list:t
            show me the list:fruit
        '''

    new_lc = []
    for i in range(len(lc)):
        new_lc.append((i, lc[i]))
    print("显示list的编号以及内容：", new_lc)
    # 显示list的编号以及内容： [(0, '苹果'), (1, 'banana'), (2, 1), (3, 2), (4, 3), (5, 'f'), (6, 'r'), (7, 'u'), (8, 'i'), (9, 't'), (10, 'fruit')]

    id = user("Alvin")
    id.showname()

    # tuple
    t_new_lc = list(enumerate(lc, start=1))
    print(t_new_lc)
    # [(1, '苹果'), (2, 'banana'), (3, 1), (4, 2), (5, 3), (6, 'f'), (7, 'r'), (8, 'u'), (9, 'i'), (10, 't'), (11, 'fruit')]
    print(type(t_new_lc[2]))  # <class 'tuple'>
    print(t_new_lc[1])  # (2, 'banana')

    # dic
    d = {'Lilei': 30, 'Hanmeimei': 29, 'Lily': 28}
    for name, age in d.items():
        print('%s is %d years old.' % (name, age))
        '''
            Lilei is 30 years old.
            Hanmeimei is 29 years old.
            Lily is 28 years old.
        '''

    # numpy
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    v = np.array([1, 0, 1])
    y = x + v
    print(y)
    '''
        [[ 2  2  4]
         [ 5  5  7]
         [ 8  8 10]
         [11 11 13]]
    '''


# 定义一个函数
def simpleCal():
    # 行末尾的\是续行符号
    print("首先学习" + "数值型的基本运算+字符串\
    ,bool类型，以及print的用法")  # 首先学习数值型的基本运算+字符串    ,bool类型，以及print的用法
    x = 5
    print(type(x))  # <class 'int'>
    print("x is %s, x+1 is %d, x*2=%d, x的平方是：%d" % (x, x + 1, x * 2, x ** 2))
    # x is 5, x+1 is 6, x*2=10, x的平方是：25
    # bool类型不但支持!=, ==,但支持and or not的表达
    if (x <= 5) == True and False:
        x += 1  # 自加不能用x++或者x--
        x *= 2  # 自乘
    elif x > 5:
        x /= 2
    x -= 2
    print("现在的X值为%s" % x)  # 现在的X值为3
    y = 3.0
    print("x/2 is:%s, y/2 is:%.3f" % (x / 2, y / 2))  # x/2 is:1.5, y/2 is:1.500
    print(type(x / 2))  # <class 'float'>
    str_break = "we will learn something about string!"
    str_tmp1 = "follow me"
    str_tmp2 = '%r %s %d' % (str_tmp1, "in", 2016)  # 格式化字符串
    print(str_break.capitalize() + "\r" + "\r" + str_tmp2.upper() + "---" * 20)
    # 'FOLLOW ME' IN 2016------------------------------------------------------------


# 定义一个类，名字为user
class user:
    def __init__(self, name):  # 构造函数
        self.name = name
        print("用户名设置完毕。")
        # 用户名设置完毕。

    def showname(self):  # 公有成员函数
        print("Current user's name is:", self.name)
        # Current user's name is: Alvin


'''
    一般在脚本最后调用主函数main（）；当我们直接运行当前脚本的时候__name__相当于__main__。
'''
if __name__ == '__main__':
    print("Welcome to July's blog!")  # Welcome to July's blog!
    # print('Welcome to July\'s blog!')
    main()

    # 展示matplot库
    img = np.zeros((100, 100))
    img[np.random.randint(0, 100, 500), np.random.randint(0, 100, 500)] = 255
    img2 = filters.gaussian_filter(img, 4, order=2)
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, img2, cmap="gray")
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    Image(buf.getvalue())
