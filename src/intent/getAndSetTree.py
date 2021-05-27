class Node():
    def __init__(self, name, father_node=None):
        self.name = name
        self.time = None
        self.effect = None
        self.character = None
        self.location = None
        self.method = None
        self.father_node = father_node

    def __str__(self):
        return self.getName()

    def getFatherNode(self):
        return self.father_node

    def setFatherNode(self, father_node):
        self.father_node = father_node

    def getName(self):
        return self.name

    def getTime(self):
        return self.time

    def getEffect(self):
        return self.effect

    def getLocation(self):
        return self.location

    def getCharacter(self):
        return self.character

    def getMethod(self):
        return self.method

    def setName(self, name):
        self.name = name

    def setTime(self, time):
        self.time = time

    def setEffect(self, effect):
        self.effect = effect

    def setLocation(self, location):
        self.location = location

    def setCharacter(self, character):
        self.character = character

    def setMethod(self, method):
        self.method = method

def initializeTree():
    """
    :param all_Node: 为True时返回意图树的所有结点，否则仅返回叶子结点
    """
    #第零层，即根节点
    node_0_0 = Node("操控国家政治")

    #第一层
    node_1_0 = Node("巩固国内政治",father_node=node_0_0)
    node_1_1 = Node("阻碍他国政治", father_node=node_0_0)
    node_1_2 = Node("其他", father_node=node_0_0)

    #第二层，即分类的意图
    node_2_0 = Node("控制暴乱",father_node=node_1_0)
    node_2_1 = Node("控制疫情_美国",father_node=node_1_0)
    node_2_2 = Node("政治斗争",father_node=node_1_0)
    node_2_3 = Node("缓解经济压力_美国",father_node=node_1_0)
    node_2_4 = Node("提高社会保障",father_node=node_1_0)
    node_2_5 = Node("军事干涉世界局势",father_node=node_1_1)
    node_2_6 = Node("操控国际经济",father_node=node_1_1)
    node_2_7 = Node("扰乱国际秩序",father_node=node_1_1)
    node_2_8 = Node("国际疫情政治化",father_node=node_1_1)
    node_2_9 = Node("借环境问题打压别国",father_node=node_1_1)

    return [node_0_0, node_1_0, node_1_1, node_1_2, node_2_0, node_2_1, node_2_2, node_2_3, node_2_4, node_2_5, node_2_6, node_2_7, node_2_8, node_2_9]



def getChainByNode(node, chain = []):
    nodeList = initializeTree()
    if node.getName() == "其他":
        chain.append(node.getFatherNode())
        chain.append(node)
    else:
        chain.append(node.getFatherNode().getFatherNode())
        chain.append(node.getFatherNode())
        chain.append(node)

    return {"treeChain":chain, "allTree":nodeList}

def getChainByName(name, chain = []):
    nodeList = initializeTree()
    leafNodeList = nodeList[3:]
    for node in leafNodeList:
        if node.getName() == name:
            result = getChainByNode(node, chain = chain)
            return result
    print("输入有误，无" + name + "意图")

if __name__ == '__main__':

    result = getChainByName("借环境问题打压别国")
    print()