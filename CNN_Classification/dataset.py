import re  # 正则表达式包
import os
import random
import tarfile  # 压缩解压包
import urllib   # 超链接包
from torchtext import data  # 文本预处理


class TarDataSet(data.Dataset):
    """
    从压缩文件中 解压文件数据集
    属性：
        超链接
        文件名
        目录名
    """
    def download_or_unzip(cls, root):
        # 路径拼接 根路径和目录名组成目标路径
        path = os.path.join(root, cls.dirname)
        # 如果不存在，就创建文件名
        print(path)
        if not os.path.isdir(path):
            print("路径不存在")
            tpath = os.path.join(root, cls.filename)
            # 判断路径是否为文件
            if not os.path.isfile(tpath):
                print("文件不存在")
                # 下载指定url内容到本地路径tpath
                print(cls.url)
                urllib.request.urlretrieve(cls.url, tpath)
            # 尝试解压文件
            with tarfile.open(tpath, 'r') as tfile:
                print("解压中")
                tfile.extractall(root)
        # 返回解压文件路径
        return os.path.join(path, '')


# 自定义Dataset类
class MR(TarDataSet):
    """
    设置MR数据集的下载连接 设置本地文件名
    """
    url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    # 名义上归类管，但实际上 可以直接调用，但不能调用类的变量
    @staticmethod
    def sort_key(ex):
        # 静态方法
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """
        给定 路径和文件名 创建一个 MR 数据集 实例
        :param text_field: 存放文本数据 X
        :param label_field: 存放标签数据 Y（监督学习）
        :param path: 存放数据的路径
        :param examples: 这个样本包含所有的数据
        :param kwargs:当你传入key=value时，存储的是字典形式，不影响参数位置
        """
        def clean_str(string):
            """
            初始化后，自动 对数据进行清理
            :param string: 传入的数据
            :return:
            """
            # re.sub(正则表达式, 替换别人的字符， 处理的字符串)
            # 清理多余字符，除了这些字符[A-Za-z0-9(),!?\'\`] 一律替换为 空格
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            # 拆分缩写，将 （this's） 替换成 （this 's）
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            # \s匹配空格，匹配两个及其以上的 空格 替换成单个空格
            string = re.sub(r"\s{2,}", " ", string)
            # string() 里面什么也没有写 表示去除首尾空格
            return string.strip()
        # 在分词之后和数值化之前使用的管道 对数据进行清理
        # 管道会对输入的数据进行转化
        text_field.preprocessing = data.Pipeline(clean_str)
        print(text_field, label_field)
        # fields 可简单理解为每一列数据和Field对象的绑定关系
        fields = [('text', text_field), ('label', label_field)]
        # 模式不存在时 就创建模式
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                # 构建 正样本
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                # 构建 负样本
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    # 修饰符 不需要实例化，可以直接调用
    # 重写 splits 自定义划分数据
    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """
        创建 数据集实例 用来划分 数据集
        :param text_field: 用来表示句子数据
        :param label_field: 用来表示标签数据
        :param dev_ratio: 训练集和测试集的比例
        :param shuffle: 是否打乱数据
        :param root: 数据集的根目录
        :param kwargs:
        :return:
        """
        # 下载数据集 并获取数据集 路径
        print("正在下载")
        path = cls.download_or_unzip(cls, root)
        # 用来表示一个样本，数据+标签
        # 构建样本
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        # 打乱数据
        if shuffle:
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))
        # li = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # print("li[-1:]: ", li[-1:])
        # print("li[:-1]: ", li[:-1])
        # [0 1 2 ... dev_index ... last]
        # 从前往后划分 划到 dev_index为止 为 第一个集 应该是训练集
        # 从后往前化， 倒着数 dev_index 划到最后一个last （即剩下的）为 测试集
        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))
