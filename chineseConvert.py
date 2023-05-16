import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

# 将会以list的形式进行转化
class strLabelConverter(object):
    def __init__(self, lbpath):
        self.lbpath = lbpath
        self.chars = self.getchars()
        self.chardic = {}
        for i in range(len(self.chars)):
            self.chardic[self.chars[i]] = i


    def getchars(self):
        with open(self.lbpath, "r") as f:
            lines = f.readlines()
        chars = [line.strip() for line in lines]
        chars[0] = " "
        return chars

    # 将text文本转化成ctc可以识别的标签, 支持单个text转化，支持多个text转化["aaa", "cccc"]
    def encode(self, text):
        if isinstance(text, str):
            lbs = [self.chardic[char] for char in text]
            length = [len(lbs)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)  # 所有的字符都合成一个大字符
            lbs, _ = self.encode(text)
        return torch.IntTensor(lbs), torch.IntTensor(length)

    # 将输出结果，返回字符；有单个字符和多个字符；单个字符显示[1,2,4], [n]; 多个字符[1,2,3,4], [1, n] 都是tensor的形式
    def decode(self, t, length):
        if length.numel() == 1:
            length = length[0]
            char_list = []
            for i in range(length):
                # if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(self.chars[t[i]])  # 第几个字符 不需要减去1
            return char_list
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l])))
                index += l
        return texts