import matplotlib.pyplot as plt
import math
import numpy as np
import random

#将城市坐标存放大cities字典中
#读取数据
f=open("test.txt")
data = f.readlines()
#将cities初始化为字典，防止下面被当成列表
cities = {}
for line in data:
    #原始数据以\n换行，将其替换掉
    line=line.replace("\n","")
    #最后一行以EOF为标志，如果读到就证明读完了，退出循环
    if(line == "EOF"):
        break
    #空格分割城市编号和城市的坐标
    city = line.split(" ")
    map(int,city)
    #将城市数据添加到cities中
    cities[eval(city[0])] = [eval(city[1]),eval(city[2])]
#print(cities)

#创建信息素矩阵, 即50*50的矩阵， 初始化信息素浓度为1
matrix = np.ones((50,50))
for i in range(50):
    matrix[i,i] = 0
#print(matrix)

#定义蚂蚁类，实现一只蚂蚁的一次遍历
class Ant:
    def __init__(self):
        #用tabu表示该蚂蚁走过的城市,规定从第0个城市开始走
        self.tabu = [np.random.randint(0,50)]  #表示该蚂蚁初始位置(列表是有序的)
        #总城市为50个，因此还剩余49个城市，用allowed列表来存放为走过的城市
        self.allowed = [city for city in range(0,50) if city not in self.tabu]
        self.nowCity = self.tabu[0] #用来记录目前所在的城市
        self.a = 2 #信息素因子ɑ
        self.b = 4 #启发函数因子𝛽
        self.rho = 0.2 #信息素挥发因子𝛽
        self.fit = 0 #完成一次路径循环后的新增信息素含量

    def next(self):
        """
        计算该蚂蚁的下一次目的地
        :return:
        """
        p = list(np.zeros(50)) #存储到达下一个城市的概率
        #分别计算概率公式中的分子和分母
        sum = 0 #用来计算p的分母
        for city in self.allowed:
            if matrix[self.nowCity][city] == 0:
                continue
            tmp = math.pow(matrix[self.nowCity][city],self.a)*math.pow((1/Ant.calc2c(city,self.nowCity)),self.b)
            sum += tmp #计算分母(分子之和)
            p[city] = tmp #将计算出的分子存储在概率表中(节省内存空间)
        for city in self.allowed: #计算概率，将概率存储在表中
            p[city] = p[city] / sum
        self.random_choose(p) #调用轮盘赌法，选择下一次到达的城市

    def random_choose(self,p_list):
        """
        轮盘赌法，选择下一次到达的城市,将概率表转换为累积概率表
        :param p_list:
        :return:
        """
        r_num = random.random()
        for i in range(0, 48):
            p_list[i + 1] += p_list[i]
        for i in range(50):
            if len(self.allowed) == 1:
                tmpCity = self.allowed[0]
                self.tabu.append(tmpCity)
                self.allowed.remove(tmpCity)
                self.nowCity = tmpCity
                break
            if (r_num < p_list[i]):  # 若累积概率q(xi)大于数组中的元素m[i]，则个体x(i)被选中
                # 因此i+1为下一步将要去的城市
                self.tabu.append(i)  # 更新走过城市列表
                self.allowed.remove(i)  # 将该城市从未走过的城市列表中删除(更新)
                self.nowCity = i  # 对目前所在城市进行更新
                # p.remove(p[i])
                break  # return
            else:
                continue

    def tour(self):
        """
        遍历所有城市
        :return:
        """
        while(self.allowed):
            self.next()
            #print(len(self.allowed))
        self.fit = Ant.calcfit(self.tabu)
        
    def updateMatrix(self):
        """
        更新信息素矩阵
        :return:
        """
        #规定将城市1-城市1直降的信息素浓度设置为0
        line = [] #line储存本次经历过的城市
        for i in range(49):
            #因为矩阵是对阵的，2-1和1-2应该有相同的值，所以两个方向都要加
            #由于列表有序，因此可以表示蚂蚁k走过的路径
            line.append([self.tabu[i],self.tabu[i+1]])
            line.append([self.tabu[i+1],self.tabu[i]])
        for i in range(0,50):
            for j in range(0,50):
                if [i,j] in line:
                    matrix[i, j] = matrix[i, j] * (1 - self.rho) + self.fit
                else:
                    matrix[i, j] = matrix[i, j] * (1 - self.rho)

    @classmethod #该方法和蚂蚁无关，因此设置为类函数
    def calcfit(cls,addr): #计算能见度(用于决定释放多少信息素)   Q/Lk(蚁周模型)
        """
        #计算适应度，即距离分之一，这里采用伪欧氏距离
        采用伪欧式距离的原因是因为: 需要符合实际
        1.地球是圆的（欧式距离计算的是平面距离）
        2.大数据不方便处理，所以采用了这个公式。
        :param addr:
        :return:
        """
        sum = 0
        addr.append(addr[0])
        for i in range(len(addr) - 1):
            nowCity = addr[i]  # 表示当前所在城市
            nextCity = addr[i + 1]  # 表示下一站城市
            nowLoc = cities[nowCity]  # 表示当前所在城市的坐标
            nextLoc = cities[nextCity]  # 表示下一站城市的坐标
            sum += math.sqrt(((nowLoc[0] - nextLoc[0]) ** 2 + (nowLoc[1] - nextLoc[1]) ** 2) / 10)
        return 20 / sum

    @classmethod
    def calc2c(cls,city1, city2):
        """
        计算俩个城市之间的距离
        :param city1:
        :param city2:
        :return:
        """
        return math.sqrt(math.pow(cities[city1][0] - cities[city2][0], 2) +
                         math.pow(cities[city1][1] - cities[city2][1], 2))

    def clear(self):
        """
        当一只蚂蚁走完一次之后，则恢复初始状态
        :return:
        """
        self.tabu = [np.random.randint(0,50)]
        self.allowed = [city for city in range(0,50) if city not in self.tabu]
        self.nowCity = self.tabu[0]
        self.fit = 0


#蚁群算法的类，实现了算法运行过程
class ACO:
    def __init__(self,initN=75):
        self.initN = initN
        self.bestTour = [i for i in range(50)]
        self.bestFit = Ant.calcfit(self.bestTour)

    def startAnt(self):
        """
        蚁群迭代函数
        :return:
        """
        i = 0
        ant = Ant() #蚂蚁群(看做)
        Gen = []  #迭代次数
        dist = []  #距离，这两个列表是为了画图
        while (i < self.initN):
            i += 1
            ant.tour()
            ant.updateMatrix()
            if (ant.fit > self.bestFit):
                self.bestFit = ant.fit
                self.bestTour = ant.tabu
            print("第{0}次遍历的路径长度是: {1}".format(i,20 / self.bestFit))
            ant.clear()
            Gen.append(i)
            dist.append(20 / self.bestFit)
        # 绘制求解过程曲线
        plt.plot(Gen, dist, '-r')
        plt.show()


if __name__ == '__main__':
    a = ACO(100)
    a.startAnt()
    print(matrix) #查看信息素矩阵
    print(a.bestTour,len(a.bestTour)) #打印最优路径，并且确定最终返回初始位置