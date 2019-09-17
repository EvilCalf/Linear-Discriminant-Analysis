"""
基于信息熵进行划分选择的决策树算法
"""
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'


class Node(object):
    '''
    definition of decision node class

    attr: attribution as parent for a new branching
    attr_down: dict: {key, value}
            key:   categoric:  categoric attr_value
                   continuous: '<= div_value' for small part
                               '> div_value' for big part
            value: children (Node class)
    label： class label (the majority of current sample labels)
    '''
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init


def TreeGenerate(DateSet):
    ''' 
    Branching for decision tree using recursion 
    决策树的生成是一个递归的过程
    @param DateSet: the pandas dataframe of the data_set 参数给一个数据集
    @return root: Node, the root node of decision tree 返回决策树的根节点
    '''
    # generating a new root node
    new_node = Node(None, None, {})
    # DateSet的倒数第一列是标签，也就是该数据集下的好坏瓜
    label_arr = DateSet[DateSet.columns[-1]]

    label_count = NodeLabel(label_arr)
    if label_count:  # assert the label_count isn's empty
        # 该节点最多数的标签即是该结点的判断结果
        new_node.label = max(label_count, key=label_count.get)

        # end if there is only 1 class in current node data
        # end if attribution array is empty
        # 如果只有一种情形或者已经没有标签了，那么决策树就已经构建好了
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node

        # get the optimal attribution for a new branching
        # 利用信息熵找出下一个合理的划分标准
        new_node.attr, div_value = OptAttr(DateSet)

        # recursion 递归
        if div_value == 0:  # categoric variable 离散属性
            value_count = ValueCount(DateSet[new_node.attr])
            for value in value_count:
                DateSet_v = DateSet[DateSet[new_node.attr].isin(
                    [value])]  # get sub set
                # delete current attribution，删除当前属性
                DateSet_v = DateSet_v.drop(new_node.attr, 1)
                # 按照新的属性依据一步步往下生成树
                new_node.attr_down[value] = TreeGenerate(DateSet_v)

        else:  # continuous variable # left and right child 连续属性
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            DateSet_v_l = DateSet[DateSet[new_node.attr] <=
                                  div_value]  # get sub set
            DateSet_v_r = DateSet[DateSet[new_node.attr] > div_value]

            new_node.attr_down[value_l] = TreeGenerate(DateSet_v_l)
            new_node.attr_down[value_r] = TreeGenerate(DateSet_v_r)

    return new_node


def Predict(root, DateSet_sample):
    '''
    make a predict based on root
    @param root: Node, root Node of the decision tree 输入根节点
    @param DateSet_sample: dataframe, a sample line 
    '''
    try:
        import re  # using Regular Expression to get the number in string
    except ImportError:
        print("module re not found")

    while root.attr != None:
        # continuous variable 连续属性
        if DateSet_sample[root.attr].dtype == float:
            # get the div_value from root.attr_down
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                break
            if DateSet_sample[root.attr].values[0] <= div_value:
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]

        # categoric variable 离散属性
        else:
            key = DateSet_sample[root.attr].values[0]
            # check whether the attr_value in the child branch
            if key in root.attr_down:
                root = root.attr_down[key]
            else:
                break

    return root.label


def NodeLabel(label_arr):
    '''
    calculating the appeared label and it's counts

    @param label_arr: data array for class labels 输入参数为一个arry
    @return label_count: dict, the appeared label and it's counts 输出为不同标签下不同的个数
    '''
    label_count = {}  # store count of label

    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    return label_count


def ValueCount(data_arr):
    '''
    calculating the appeared value for categoric attribute and it's counts

    @param data_arr: data array for an attribute
    @return value_count: dict, the appeared value and it's counts
    '''
    value_count = {}  # store count of value

    for label in data_arr:
        if label in value_count:
            value_count[label] += 1
        else:
            value_count[label] = 1

    return value_count


def OptAttr(DateSet):
    '''
    find the optimal attributes of current data_set
    找到最合适的下一个分类的属性
    @param DateSet: the pandas dataframe of the data_set 
    @return opt_attr:  the optimal attribution for branch
    @return div_value: for discrete variable value = 0
                    for continuous variable value = t for bisection divide value
    '''
    info_gain = 0
    # 遍历除了标签以外的所有属性，找到信息增益最大的分类属性
    for attr_id in DateSet.columns[1:-1]:
        info_gian_tmp, div_value_tmp = InfoGain(DateSet, attr_id)
        if info_gian_tmp > info_gain:
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp

    return opt_attr, div_value


def InfoGain(DateSet, index):
    '''
    calculating the information gain of an attribution
    计算属性的信息增益
    @param DateSet:      dataframe, the pandas dataframe of the data_set 参数之一数据集
    @param attr_id: the target attribution in DateSet                    参数之二属性
    @return info_gain: the information gain of current attribution       输出当前划分的信息增益
    @return div_value: for discrete variable, value = 0                  输出之二划分权值
                for continuous variable, value = t (the division value)
    '''
    info_gain = InfoEnt(
        DateSet.values[:, -1])  # info_gain for the whole label 得出当前节点的信息熵
    div_value = 0  # div_value for continuous attribute

    n = len(DateSet[index])  # the number of sample
    # 1.for continuous variable using method of bisection（连续变量）
    if DateSet[index].dtype == float:
        sub_info_ent = {}  # store the div_value (div) and it's subset entropy

        DateSet = DateSet.sort_values([index],
                                      ascending=1)  # sorting via column 由小到大排序
        DateSet = DateSet.reset_index(drop=True)  # 重新设置索引加上序号

        data_arr = DateSet[index]  # 提取当前属性下那一列数据
        label_arr = DateSet[DateSet.columns[-1]]  # 提取最后评估结果标签
        # 采用二分法对连续属性进行处理，遍历n-1次求出一个信息熵数组
        for i in range(n - 1):
            div = (data_arr[i] + data_arr[i + 1]) / 2
            sub_info_ent[div] = ((i + 1) * InfoEnt(label_arr[0:i + 1]) / n) \
                                + ((n - i - 1) * InfoEnt(label_arr[i + 1:-1]) / n)
        # our goal is to get the min subset entropy sum and it's divide value，选出信息熵最小的，也就是信息增益最大的
        div_value, sub_info_ent_max = min(sub_info_ent.items(),
                                          key=lambda x: x[1])
        info_gain -= sub_info_ent_max

    # 2.for discrete variable (categoric variable)（离散变量）
    else:
        data_arr = DateSet[index]
        label_arr = DateSet[DateSet.columns[-1]]
        value_count = ValueCount(data_arr)
        # 遍历所有当前属性下所有情形，最终求出信息增益
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n

    return info_gain, div_value


def InfoEnt(label_arr):
    '''
    calculating the information entropy of an attribution
    计算信息熵
    @param label_arr: ndarray, class label array of data_arr
    @return ent: the information entropy of current attribution
    '''
    try:
        from math import log2
    except ImportError:
        print("module math.log2 not found")

    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)

    for key in label_count:
        ent -= (label_count[key] / n) * log2(label_count[key] / n)

    return ent


def DrawPNG(root, out_file):
    '''
    visualization of decision tree from root.
    @param root: Node, the root node for tree.
    @param out_file: str, name and path of output file
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    g = graphviz.Dot()  # generation of new dot

    TreeToGraph(0, g, root)
    g2 = graphviz.graph_from_dot_data(g.to_string())

    g2.write_png(out_file)


def TreeToGraph(i, g, root):
    '''
    build a graph from root on
    @param i: node number in this tree
    @param g: pydotplus.graphviz.Dot() object
    @param root: the root node

    @return i: node number after modified
#     @return g: pydotplus.graphviz.Dot() object after modified
    @return g_node: the current root node in graphviz
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label, fontname="FangSong"))

    for value in list(root.attr_down):
        i, g_child = TreeToGraph(i + 1, g, root.attr_down[value])
        g.add_edge(
            graphviz.Edge(g_node, g_child, label=value, fontname="FangSong"))

    return i, g_node