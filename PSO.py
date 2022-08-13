import copy
import numpy as np
import matplotlib.pyplot as plt
import random
"""
使用粒子群算法解决TSP问题。(无边界问题)
粒子群算法解决的俩类问题：
1.具有连续型的变量，例如：求函数最值问题，也称为连续优化问题。
2.具有离散的变量，例如：TSP问题，成为组合优化问题。
"""
class City:
    def __init__(self, city_num=15):
        self.city_num = city_num #城市数量
        self.city_pos_list = np.random.rand(self.city_num, 2) #城市坐标
        self.city_dist_mat = self.build_dist_mat(self.city_pos_list) #城市距离矩阵

    #构建城市距离矩阵(采用欧式距离进行计算)
    def build_dist_mat(self,input_list:np.ndarray) -> np.ndarray:
        num_city = self.city_num #城市数量
        dist_mat = np.zeros((num_city,num_city)) #初始化距离矩阵(全为0)
        for i in range(num_city):
            for j in range(i+1,num_city):
                #储存城市坐标上每个维度的差值
                d = input_list[i,:] - input_list[j,:]
                #计算城市之间的距离矩阵
                dist_mat[i, j] = np.sqrt(np.dot(d, d.T)) #矩阵的乘法
                #由于距离矩阵是对称矩阵,所以矩阵关于主对角线对称
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat

#创建一个城市类的对象
city = City()


#创建粒子个体类
class Individual:
    #对粒子进行初始化
    def __init__(self, city_list=None):
        self.city_len = city.city_num #单个粒子经过的城市个数
        if city_list is None:
            city_list = [x for x in range(self.city_len)]
            random.shuffle(city_list) #打乱顺序，使其具有随机性
        self.city_list = city_list #单个粒子的路径
        self.fitness = Individual.fitness_func(self.city_list)

    #计算粒子的适应度(即粒子经过的总路程)
    @staticmethod
    def fitness_func(input_list:list) -> float:
        fitness = 0 #用来表示最终的适应度(单个粒子所走的路程)
        for i in range(len(input_list)-1):
            now_city = input_list[i] #当前所处城市
            next_city = input_list[i+1] #下一站要去的城市
            #计算城市之间的距离，即距离越短，则适应度越高
            fitness += city.city_dist_mat[now_city, next_city]
        # 连接首尾，即加上首尾城市之间的距离
        fitness += city.city_dist_mat[input_list[-1], input_list[0]]
        return fitness


#定义粒子群算法类
"""
在TSP问题中优化的是序列中访问的顺序，上一次的速度(在TSP问题中即交换子)的参考意义不大，
因此为了降低问题的复杂度，则将Vi定义为一个空的交换序列。
"""
class PSO:
    def __init__(self):
        self.iter_max = 500 #最大迭代次数
        self.individual_num = 50 #粒子个数(即种群中个体数目)
        self.r1 = 0.7 #pbest-xi的保留概率(用来增加搜索的随机性)
        self.r2 = 0.8 #gbest-xi的保留概率
        self.c1 = 1 #个体学习因子
        self.c2 = 1 #社会学习因子
        self.city_num = city.city_num #城市数量
        self.distance_matrix = city.city_dist_mat #城市距离矩阵
        self.particles = [] #初始化后的粒子种群
        self.g_best_fitness_list = []  # 用来存储迭代过程对应的最小适应度(g_best对应适应度)
        self.g_best = []  # 最优解序列(g_best)

    #定义种群初始化函数
    def group_init(self) -> None:
        #初始化(随机生成初代种群)
        self.particles = [Individual() for _ in range(self.individual_num)]

    #定义速度更新函数
    @staticmethod
    def update_ss_speed(x_best, x_i, r) -> list:
        """
        计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(p_best-xi) 和 r2(g_best-xi)
        即p_best-xi则表示有一个交换序列ss，它使得xi经过ss交换后得到p_best, g_best同理。
        Vi(t+1) = c1*r1*(p_best-Xi) + c2*r2(g_best-Xi)
        :param x_best: 存储最优粒子元素, x_best: p_best 或者 g_best
        :param x_i: 粒子当前的解
        :param r: 随机因子(即：保留概率)
        :return: 返回的是交换子序列
        """
        velocity_ss = [] #用来存储交换子序列
        for i in range(len(x_i)):
            if x_i[i] != x_best[i]: #判断当前粒子是否为最优粒子元素
                temp = list(x_best).index(x_i[i]) #获取当前粒子和最优粒子之间不同元素所在的索引
                #temp = np.where(x_i == x_best[i])[0][0]
                so = (i, temp, r) #获取交换子
                velocity_ss.append(so) #加入交换序列
                x_i[i], x_i[temp] = x_i[temp], x_i[i] #对序列进行更新
        return velocity_ss

    #定义位置更新函数(交换子看做速度，生成的新序列则代表粒子的位置发生变化)
    @staticmethod
    def update_ss_location(x_i, ss) -> list:
        """
        Xi(t+1) = Xi(t) + Vi(t+1)
        :param x_i: 表示粒子当前的解
        :param ss: 用来存储交换子序列,由交换子组成的交换序列
        :return: 返回序列xi解
        """
        for i, j, r in ss:
            random_num = random.random() ##随机生成0-1的一个数,用来表示随机因子
            if random_num <= r: #有一定的概率会更新位置
                x_i[i], x_i[j] = x_i[j], x_i[i]
        return x_i #返回粒子更新后的解

    #获取最终的g_best
    def get_g_best(self):
        self.g_best.append(self.g_best[0])

    #运行函数
    def train(self):
        """
        g_best: 表示表示群体的历史最佳位置
        p_best: 表示个体粒子的历史最佳位置
        :return:
        """
        #对粒子群进行初始化
        self.group_init()
        #初始化p_best和g_best以及对应的个体历史最优适应度和群体最优适应度
        xx = np.zeros((self.individual_num,self.city_num),np.int)
        p_best_fitness_list = np.zeros((self.individual_num,1))
        for index in range(self.individual_num):
            xx[index] = self.particles[index].city_list
            p_best_fitness_list[index] = self.particles[index].fitness
        p_best_list = xx #初始化粒子最优解
        g_best = p_best_list[p_best_fitness_list.argmin()] #沿着指定轴返回最小值对应的索引
        g_best_fitness = p_best_fitness_list.min() #返回沿给定轴的最小值。
        #记录算法迭代效果
        self.g_best_fitness_list.append(g_best_fitness)
        #开始迭代
        for i in range(self.iter_max-1):
            #开始遍历种群
            for index in range(self.individual_num):
                p_best_i = copy.deepcopy(p_best_list[index])
                x_i = copy.deepcopy(xx[index]) #粒子当前解
                # 计算交换序列，即 v = r1(p_best-xi) + r2(g_best-xi)
                ss_1 = PSO.update_ss_speed(p_best_i, x_i, self.r1)  # 计算p_best交换子序列
                ss_2 = PSO.update_ss_speed(g_best, x_i, self.r2)  # 计算g_best交换子序列
                ss = ss_1 + ss_2  # 将俩个列表合并成一个列表,ss_1.extend(ss_2)
                # 执行交换操作，即 x = x + v
                x_i = PSO.update_ss_location(x_i, ss)
                #判断是否为粒子最优解
                p_fitness_new = Individual.fitness_func(x_i)
                p_fitness_old = p_best_fitness_list[index]
                if p_fitness_new < p_fitness_old:
                    p_best_fitness_list[index] = p_fitness_new
                    p_best_list[index] = x_i
                #判断是否为全局最优
                g_best_fitness_new = p_best_fitness_list.min()
                g_best_new = p_best_list[p_best_fitness_list.argmin()]
                if g_best_fitness_new < g_best_fitness:
                    g_best_fitness = g_best_fitness_new
                    g_best = g_best_new
            self.g_best_fitness_list.append(g_best_fitness)
            self.g_best = list(g_best)

    #画图
    def draw_picture(self):
        self.train()
        self.get_g_best()
        result_pos_list = city.city_pos_list[self.g_best, :]  # 获取城市坐标
        # 绘图
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # 最短路线图
        plt.figure()
        plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
        plt.title("路线")
        # plt.legend()
        plt.show()
        # 适应度收敛曲线图
        plt.figure()
        plt.plot(self.g_best_fitness_list)
        plt.title("适应度曲线")
        # plt.legend()
        plt.show()

if __name__ == "__main__":
    pso = PSO()
    pso.draw_picture()
    #输出最优路径
    print(pso.g_best)
