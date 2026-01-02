import torch
from torch.utils.data import Dataset

class MIDD(Dataset):
    def __init__(self, dataset_name, response_texts, labels, tokenizer, max_length=64):
        """
        初始化数据集
        Args:
            response_texts (list of list): 每个被试的响应文本列表
            labels (list): 每个被试的标签(0-非抑郁, 1-抑郁)
            tokenizer (AutoTokenizer): 用于文本编码的分词器
            max_length (int): 最大序列长度
        """
        self.dataset_name = dataset_name
        self.response_texts = response_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.stimulus_texts = [
            "如果你有一个假期可以去旅行，请分享一下你的旅行计划。", "请分享一段你最喜欢的回忆，大致描述一下场景。", "最近情绪怎么样？这种情绪对你的生活有什么影响？",
            "最近身体状况如何？对你的生活有什么影响？", "你如何评价你自己？", "请告诉我一个让你感觉不好的经历，或者描述一个让你非常痛苦的事件。",
            "这个问题是一个正面情绪人脸图片描述", "这个问题是一个中性情绪人脸图片描述", "这个问题是一个负面情绪人脸图片描述"
        ]   
        
        # 预处理刺激文本
        self.stimulus_encodings = tokenizer(
            self.stimulus_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(self.stimulus_encodings['input_ids'].shape)
        
        # 刺激类型标记：0-中性，1-正性，2-负性
        self.stimulus_types = torch.tensor([1, 1, 0, 0, 0, 2, 1, 0, 2])  # 固定编码
        self.stimulus_positions = torch.arange(len(self.stimulus_texts))  # 刺激位置编码

    def __len__(self):
        """返回数据集大小"""
        return len(self.response_texts)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本，包括刺激数据、响应数据和标签
        
        Args:
            idx (int): 样本索引
        Returns:
            dict: 包含刺激数据、响应数据和标签的字典
        """
        # 获取当前被试的响应文本
        responses = self.response_texts[idx]
        
        # 编码响应文本
        response_encodings = self.tokenizer(
            responses,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 构建返回数据
        stimuls_data = {
            'input_ids': self.stimulus_encodings['input_ids'],  # [9, max_len]
            'attention_mask': self.stimulus_encodings['attention_mask'],  # [9, max_len]
            'types': self.stimulus_types,  # [9]
            'positions': self.stimulus_positions  # [len(s)]
        }
        
        response_data = {
            'input_ids': response_encodings['input_ids'],  # [9, max_len]
            'attention_mask': response_encodings['attention_mask'],  # [9, max_len]
            'stimuls_type': stimuls_data['types'].clone().detach()
        }

        label = torch.tensor(self.labels[idx], dtype=torch.float)  # 标签
        return stimuls_data, response_data, label 


class MODMA(Dataset):
    def __init__(self, dataset_name, response_texts, labels, tokenizer, max_length=64):
        """
        初始化数据集
        Args:
            response_texts (list of list): 每个被试的响应文本列表
            labels (list): 每个被试的标签(0-非抑郁, 1-抑郁)
            tokenizer (AutoTokenizer): 用于文本编码的分词器
            max_length (int): 最大序列长度
        """
        self.dataset_name = dataset_name
        self.response_texts = response_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.stimulus_texts = [
            "你平时喜欢看什么电视节目？", "你有什么兴趣爱好?", "最近有什么让你高兴的事情？",
            "如果有一个假期可以去旅游，你想去什么地方？", "你收到过最好的礼物是什么？", "你最美好的回忆是什么?",
            "你最近情绪怎么样?", "你最近身体状况怎么样?", "你和家里人关系怎么样?",
            "能介绍一下你最好的朋友吗?", "你对未来三年有什么计划？", "你如何评价自己?",
            "当你和朋友发生矛盾时，你有什么感受或处理方式？", "当你晚上睡不着觉时，你会做什么？", "当你和家人发生不愉快时，你会做什么？",
            "分享一段你觉得糟糕的经历。", "什么情况让你感到绝望？当你感到绝望时，你会怎么做？", "你怎么看待自杀？"
        ]   
        
        # 预处理刺激文本
        self.stimulus_encodings = tokenizer(
            self.stimulus_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(self.stimulus_encodings['input_ids'].shape)
        
        # 刺激类型标记：0-中性，1-正性，2-负性
        self.stimulus_types = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2, 0, 2, 2, 2, 2])  # 固定编码
        self.stimulus_positions = torch.arange(len(self.stimulus_texts))  # 刺激位置编码

    def __len__(self):
        """返回数据集大小"""
        return len(self.response_texts)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本，包括刺激数据、响应数据和标签

        Args:
            idx (int): 样本索引
        Returns:
            dict: 包含刺激数据、响应数据和标签的字典
        """
        # 获取当前被试的响应文本
        responses = self.response_texts[idx]
        
        # 编码响应文本
        response_encodings = self.tokenizer(
            responses,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 构建返回数据
        stimuls_data = {
            'input_ids': self.stimulus_encodings['input_ids'],  # [18, max_len]
            'attention_mask': self.stimulus_encodings['attention_mask'],  # [18, max_len]
            'types': self.stimulus_types,  # [18]
            'positions': self.stimulus_positions  # [len(s)]
        }
        
        response_data = {
            'input_ids': response_encodings['input_ids'],  # [18, max_len]
            'attention_mask': response_encodings['attention_mask'],  # [18, max_len]
            'stimuls_type': stimuls_data['types'].clone().detach()
        }

        label = torch.tensor(self.labels[idx], dtype=torch.float)  # 标签
        return stimuls_data, response_data, label    
