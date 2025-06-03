# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：main.py -> 画图
@IDE                ：PyCharm
@Author             ：王勉
@Date               ：2024/3/6 14:29
@email              ：wangmian33@qq.com
==================================================
"""
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def crop_top(image_path, output_path):
    # 打开图片
    img = Image.open(image_path)

    # 获取图片尺寸
    width, height = img.size

    # 计算要裁剪的上方的高度
    crop_height = int(height / 6)

    # 裁剪图片
    cropped_img = img.crop((0, crop_height, width, height))

    # 保存裁剪后的图片
    cropped_img.save(output_path)

def fig_snaps_datasets(data_name,save_name,y1,y2,y3,y_begin, y_end):
    # 生成数据
    snap1 = len(y1)
    snap2 = len(y2)
    snap3 = len(y3)
    x1 = np.arange(1, snap1+1, 1)
    x2 = np.arange(1, snap2+1, 1)
    x3 = np.arange(1, snap3+1, 1)
    # y1 = np.linspace(0.9, 1, 5)
    # y2 = np.linspace(0.9, 1, 5) + 0.02
    # y3 = np.linspace(0.9, 1, 5) - 0.02
    fig, ax = plt.subplots(figsize=(8, 6))
    # 画折线图
    line1,=ax.plot(x1, y1, label='AUC at 10% anomaly', color='blue', marker='o')
    line2,=ax.plot(x2, y2, label='AUC at 5% anomaly', color='green', marker='s')
    line3,=ax.plot(x3, y3, label='AUC at 1% anomaly', color='red', marker='^')

    # 设置坐标轴范围和精度
    plt.ylim(0.9, 1)
    plt.yticks(np.arange(y_begin-0.01, y_end+0.01, 0.01))
    plt.xticks(np.arange(1, snap1+1, 1))
    # 添加图例
    ax.legend(bbox_to_anchor=(0.5, 1.35), mode='expand',ncol=3,handlelength=2,frameon=0, borderpad=8)

    # 去除右边和上边的坐标系边框
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # 添加网格
    # plt.grid(True)

    # 添加标题和标签
    plt.title(data_name)
    plt.xlabel('snapshot')
    plt.ylabel('AUC')
    plt.savefig(save_name+'.png', dpi=200, bbox_inches='tight')
    # 显示图形
    plt.show()

def fig_parm_3d():
    # 创建数据
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 20, 15, 25, 30])
    z = np.array([5, 10, 8, 12, 15])

    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 画三维柱状图
    ax.bar(x, z, y, zdir='y', color='b', alpha=0.5)

    # 设置轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 显示图形
    plt.show()


def draw_3dface():

    # 创建网格数据
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2 + 1

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置颜色映射
    colors = plt.cm.Blues((Z - Z.min()) / (Z.max() - Z.min()))

    # 绘制曲面
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)

    # 隐藏坐标轴
    ax.set_axis_off()
    plt.savefig('3dface.png', dpi=300, bbox_inches='tight')
    # 显示图形
    plt.show()


if __name__ == '__main__':
    # y1 = [0.9813432835820897, 0.9632059156777386, 0.9318363491025362, 0.9505597014925373, 0.9537837676665275]
    # y2 = [0.971537558685446, 0.963121260227134, 0.922207514509724, 0.93994341563786, 0.9320305862361938]
    # y3 = [1.0, 0.982102908277405, 0.9290158371040724, 0.9313944817300521, 0.959049959049959]
    # y_begin = 0.91
    # y_end = 1.0
    # data_name = 'email-DNC'
    # save_name = data_name + 'linePlot'
    # fig_snaps_datasets(data_name,save_name,y1,y2,y3,y_begin,y_end)
    # crop_top(save_name+'.png',save_name+'.png')
    draw_3dface()