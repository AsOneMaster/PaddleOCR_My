#学生：何迅
#创建时间：2021/12/10 14:34

f = open('字符识别率text.txt')
data = f.readlines()  #逐行读取txt并存成list。每行是list的一个元素，数据类型为str
# print(data)
# l = []
# for i in range(len(data)):  #len(data)为数据行数
#     for j in range(len(list(data[0].split()))):   #len(list(data[0].split()))为数据列数
#         l.append(data[i].split('\n')[j])
# print(l)
a = 0.0
i = 0
for line in data:
    line = line.strip('\n').split(':')
    a +=float(line[1])
    i +=1

print(i)
print(float(a/i))

