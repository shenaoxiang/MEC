import numpy as np
import matplotlib.pyplot as plt
import random
import copy


#城市类
class City:
    def __init__(self,city_num=15):
        self.city_num = city_num #城市数量
        self.city_pos_list = np.random.rand(self.city_num,2) #城市坐标
        self.city_dist_mat = self.build_dist_mat(self.city_pos_list) #城市距离矩阵

    #构建城市距离矩阵
    def build_dist_mat(self, input_list:np.ndarray) -> np.ndarray:
        n = self.city_num
        dist_mat = np.zeros([n,n]) #初始化距离矩阵，全为0
        for i in range(n):
            for j in range(i+1,n):
                d = input_list[i,:] - input_list[j,:] #存储每个维度上的差值(矩阵减法)
                #计算距离的平方矩阵
                dist_mat[i,j] = np.sqrt(np.dot(d,d.T)) #点积
                #矩阵是对称矩阵
                dist_mat[j,i] =  dist_mat[i,j]
        return dist_mat

#创建一个城市实例
city = City()


#创建基因个体类
class Individual:
    def __init__(self,genes = None):
        #随机生成序列，即遍历城市的顺序为该个体的基因序列
        self.gene_len = city.city_num #基因序列长度(即城市数量)
        if genes is None:
            genes = [i for i in range(self.gene_len)]
            random.shuffle(genes) #打乱顺序
        self.genes = genes #每个个体的基因
        self.fitness = self.evaluate_fitness()

    # 计算个体基因的适应度,即用来判断个体优劣
    def evaluate_fitness(self) -> float:
        fitness = 0 #用来表示最终适应度
        for i in range(len(self.genes)-1):
            now_city = self.genes[i] #此时所处城市
            next_city = self.genes[i+1]  #下一站到达城市
            #计算城市之间的距离，距离越短适应度越强
            fitness += city.city_dist_mat[now_city,next_city] #计算遍历城市的距离
        #连接首尾，即加上首尾城市之间的距离
        fitness += city.city_dist_mat[self.genes[-1],self.genes[0]]
        return fitness


#创建遗传算法类(即种群类)
class GA:
    def __init__(self):
        self.individual_num = 60 #种群中个体数量
        self.gene_num = 400 #迭代次数
        self.cross_prob = 0.7 #交叉概率
        self.mutate_prob = 0.25 #变异概率
        self.gene_len = city.city_num  # 基因序列长度(即城市数量)
        self.best = None  # 每一代的最佳个体
        self.individual_list = []  # 每一代的个体列表
        self.result_list = []  # 每一代对应的解
        self.fitness_list = []  # 每一代对应的适应度

    #选择(复制),精英产生精英
    def select(self) -> None:
        #锦标赛选择算法(轮盘赌法也可以，不唯一),属于放回抽样
        group_num = 10 #锦标赛选择法所分的小组数
        group_size = 10 #每小组的个体数
        group_winner = self.individual_num // group_num #规定每个小组获取的人数
        winners = [] #用来储存锦标赛结果
        #10场比赛
        for i in range(group_num):
            group = [] #用来存储每场比赛的结果
            #一场比赛中的个体比拼
            for j in range(group_size):
                #随机生成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes) #个体初始化
                group.append(player)
            group = GA.rank(group)
            #取出每组的前六名，即每个小组的获胜者
            winners += group[:group_winner]
        self.individual_list = winners

    #交叉。交叉：1和2,3和4，以一定概率决定是否交叉。若交叉，则二者选择随机一个段进行交叉
    def cross(self) -> [Individual]:
        new_gene = [] #用于存储交叉基因后的新个体
        random.shuffle(self.individual_list) #将种群顺序打乱，避免随机性
        for i in range(0,self.individual_num,2):
            random_prob = random.random() #随机生成0-1的一个数,用来表示随机概率
            #父代基因,为了避免交叉的结果在原有种群基础上进行修改，这里采用深拷贝
            genes_father = copy.deepcopy(self.individual_list[i].genes) #父亲
            genes_mather = copy.deepcopy(self.individual_list[i+1].genes) #母亲
            index1 = random.randint(0, self.gene_len-2)  #交叉基因片段起始位置(index1<index2)
            index2 = random.randint(index1, self.gene_len-1) #交叉基因片段终止位置
            #记录初试基因片段在基因序列中的位置
            pos1_recorder = {value: idx for idx, value in enumerate(genes_father)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes_mather)}
            if random_prob < self.cross_prob:
                #交叉
                for j in range(index1, index2):
                    value1, value2 = genes_father[j], genes_mather[j]
                    pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1] #获取基因位置
                    #交叉基因(交换基因位置),pos1为mather基因序列中value在father基因序列中的位置,反之亦然。
                    genes_father[j], genes_father[pos1] = genes_father[pos1], genes_father[j]
                    genes_mather[j], genes_mather[pos2] = genes_mather[pos2], genes_mather[j]
                    pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                    pos2_recorder[value1], pos2_recorder[value2] = j, pos2
                new_gene.append(Individual(genes_father))
                new_gene.append(Individual(genes_mather))
            else:
                new_gene.append(Individual(genes_father))
                new_gene.append(Individual(genes_mather))
        return new_gene

    #变异
    def mutate(self,new_gene:[Individual]) -> None:
        for individual in new_gene: #遍历交叉种群列表
            temp_prob = random.random() #生成一个随机数
            if temp_prob < self.mutate_prob: #有0.25的概率会发生基因变异
                #翻转切片(若变异，取反即可)
                temp_gene = copy.deepcopy(individual.genes) #用来保存个体基因
                #生成基因片段，并且翻转(即取反)
                index_start = random.randint(0, self.gene_len-2) #起始位置
                index_end = random.randint(index_start,self.gene_len-1) #终止位置
                mutate_gene = temp_gene[index_start:index_end] #获取基因片段
                mutate_gene.reverse() #翻转
                individual.genes = temp_gene[0:index_start] + mutate_gene + temp_gene[index_end:] #拼接
            else:
                continue
        #形成新的一代
        self.individual_list = new_gene


    #排序算法(类函数)
    @classmethod
    def rank(cls, group:[Individual]) -> [Individual]:
        #自写排序算法，根据适应度值大小从小到大进行排序(此处采用冒泡排序)
        for i in range(len(group)-1):
            for j in range(len(group)-1-i):
                if group[j].fitness > group[j+1].fitness:
                    group[j], group[j+1] = group[j+1], group[j]
        return group

    #获取下一代
    def next_gene(self) -> None:
        #选择
        self.select()
        #交叉
        new_gene = self.cross()
        #变异
        self.mutate(new_gene)
        #获取新一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    #遗传
    def train(self):
        #初始化(随机生成初代种群)
        self.individual_list = [Individual() for i in range(self.individual_num)]
        self.best = self.individual_list[0] #默认情况下假设第一个个体为最优个体
        #开始迭代
        for i in range(self.gene_num):
            self.next_gene() #产生下一代
            #由于最终要返回到起始位置，因此需要连接首尾位置(基因序列中没有连接)
            result = copy.deepcopy(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)  #城市顺序(最优子代)
            self.fitness_list.append(self.best.fitness) #适应度
        return self.result_list, self.fitness_list

    #绘图函数
    def draw_picture(self):
        result_list, fitness_list = self.train()
        result = result_list[-1]
        result_pos_list = city.city_pos_list[result, :]
        # 绘图
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.figure()
        plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
        plt.title(u"路线")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(fitness_list)
        plt.title(u"适应度曲线")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ga = GA()
    ga.draw_picture()