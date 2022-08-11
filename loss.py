#学生：何迅
#创建时间：2022/5/19 9:41
#导入要使用的科学计算包numpy,pandas,可视化    matplotlib,seaborn,以及机器学习包sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from IPython.display import display
import scipy.signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plt.style.use("fivethirtyeight")
loss1_df=pd.read_csv('E:/OCRData/log/visualdl-scalar-TRAIN_loss/mv3.csv')
loss2_df=pd.read_csv('E:/OCRData/log/visualdl-scalar-TRAIN_loss/res18.csv')
loss3_df=pd.read_csv('E:/OCRData/log/visualdl-scalar-TRAIN_loss/res50.csv')
x1 = loss1_df['id']
y1 = loss1_df['value']
x2 = loss2_df['id']
y2 = loss2_df['value']
x3 = loss3_df['id']
y3 = loss3_df['value']
fig = plt.figure(figsize = (10,7))       #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
tmp1 = scipy.signal.savgol_filter(y1, 99, 1, mode= 'nearest')
tmp2 = scipy.signal.savgol_filter(y2, 99, 1, mode= 'nearest')
tmp3 = scipy.signal.savgol_filter(y3, 99, 1, mode= 'nearest')
# pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
plt.rcParams.update({'font.size': 15})
plt.tick_params(labelsize=15)
# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
p1 = pl.plot(x1,tmp1,'r-',linewidth=1.0, label = u'MobileNetv3')
pl.legend()
#显示图例
p2 = pl.plot(x2,tmp2, 'b--',dashes=(5, 2),linewidth=1.0, label = u'ResNet18')
pl.legend()
p3 = pl.plot(x3,tmp3, 'g-.',dashes=(5, 7),linewidth=1.0, label = u'ResNet50')
pl.legend()

pl.xlabel(u'iters',fontsize=20)
pl.ylabel(u'loss',fontsize=20)
# plt.title('Compare loss for different Backbone in training',fontsize=20)
# # plot the box
# tx0 = 0
# tx1 = 4000
# #设置想放大区域的横坐标范围
# ty0 = 0
# ty1 = 2
# #设置想放大区域的纵坐标范围
# sx = [tx0,tx1,tx1,tx0,tx0]
# sy = [ty0,ty0,ty1,ty1,ty0]
# pl.plot(sx,sy,"purple",linewidth=2.0)
# axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
# #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
# axins.plot(x1,tmp1 , color='red', ls='-',linewidth=1.0)
# axins.plot(x2,tmp2 , color='blue', ls='-',linewidth=1.0)
# axins.plot(x3,tmp3, color='green', ls='-',linewidth=1.0)
# axins.axis([0,4000,0,2])
#
plt.savefig("train_results_loss1.png")
pl.show()