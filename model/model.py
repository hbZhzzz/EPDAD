import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), 'code'))

import model.config as config 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    # 这里的mask是umask，表示序列的长度，有效序列长度会被填充为1，无效序列为0
    # mask_会与log_pred 和 target做矩阵乘法，将无效序列的位置置为0，这样就不会计算无效位置的损失
    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss

class Local_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Local_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep
    

class Global_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Global_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

def compute_mean_features(features, mask=None):
    """计算特征的平均值，考虑mask"""
    if mask is not None:
        # mask: [batch_size, max_num]
        features = features * mask.unsqueeze(-1)
        return features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    return torch.mean(features, dim=1)  # [batch_size, hidden_size]

def compute_variance(features_list):
    """计算不同情绪特征之间的方差"""
    # features_list: list of tensors, each [batch_size, hidden_size]
    features_stack = torch.stack(features_list, dim=1)  # [batch_size, 3, hidden_size] 3是因为只有正中负3类情绪，
    # print(f'features_stack:{features_stack.shape}') # [bs, 3, 768]
    return torch.var(features_stack, dim=1).mean(dim=-1)  # [batch_size]

def compute_adaptability(emo_features, stim_features):
    """计算情绪适应性（与刺激的匹配度）"""
    # 计算余弦相似度
    sim = F.cosine_similarity(emo_features, stim_features, dim=-1)  # [batch_size]
    return sim

class ContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature
    
    def forward(self, positive_emo, neutral_emo, negative_emo,
                pos_mask=None, neu_mask=None, neg_mask=None):
        batch_size = positive_emo.size(0)

        # 对每个情绪特征进行池化，得到一个全局表示
        positive_emo = compute_mean_features(positive_emo, pos_mask) # [batch_size, hidden_size]
        neutral_emo = compute_mean_features(neutral_emo, neu_mask) # [batch_size, hidden_size]
        negative_emo = compute_mean_features(negative_emo, neg_mask) # [batch_size, hidden_size]
        
        # 将所有特征拼接在一起
        features = torch.cat([positive_emo, neutral_emo, negative_emo], dim=0)  # [3*batch_size, hidden_size]

        # 计算完整的相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [3*batch_size, 3*batch_size]
        
        similarity_matrix = similarity_matrix / self.temperature

        # 创建标签：同一情绪类别内的样本为正例
        # labels = torch.cat([
        #                     torch.zeros(batch_size),
        #                     torch.ones(batch_size),
        #                     2 * torch.ones(batch_size)
        #                 ]).to(positive_emo.device).long()
        labels = torch.cat([torch.arange(batch_size) for _ in range(3)], dim=0)
        labels = labels.to(positive_emo.device)

        # 创建mask：同一情绪类别内的样本相似度应该大，不同情绪类别的样本相似度应该小
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # 去除对角线上的自身相似度
        mask = mask.fill_diagonal_(0)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 计算每个样本的损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 最终损失
        loss = -mean_log_prob_pos.mean()
        
        return loss
        


class DataAugmentation(nn.Module):
    def __init__(self, noise_scale=0.1, dropout_prob=0.1):
        super().__init__()
        self.noise_scale = noise_scale
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        # 添加高斯噪声
        noise = torch.randn_like(x) * self.noise_scale
        x = x + noise
        
        # 随机dropout
        x = self.dropout(x)
        
        return x


class EmotionalConsistencyLoss(torch.nn.Module):
    # 在论文里叫情绪差异感知损失，在这里还是沿用一开始一致性的说法，不再修改代码了
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, positive_emo, neutral_emo, negative_emo,
               positive_features, neutral_features, negative_features,
               labels, pos_mask=None, neu_mask=None, neg_mask=None):
       
    
        # 计算平均刺激情绪表征
        pos_emo_mean = compute_mean_features(positive_emo, pos_mask)
        neu_emo_mean = compute_mean_features(neutral_emo, neu_mask)
        neg_emo_mean = compute_mean_features(negative_emo, neg_mask)
        
        # 计算平均响应情绪表征
        pos_feat_mean = compute_mean_features(positive_features, pos_mask)
        neu_feat_mean = compute_mean_features(neutral_features, neu_mask)
        neg_feat_mean = compute_mean_features(negative_features, neg_mask)
        
        # 计算情绪方差（用于抑郁组）
        print(f'positive_features:{positive_features.shape}, pos_feat_mean:{pos_feat_mean.shape}')
        print(f'neutral_features:{neutral_features.shape}, neu_feat_mean:{neu_feat_mean.shape}')
        print(f'negative_features:{negative_features.shape}, neg_feat_mean:{neg_feat_mean.shape}')

        variance_depression = compute_variance([pos_feat_mean, neu_feat_mean, neg_feat_mean])
        
        # 计算情绪适应性（用于正常组）
        adaptability_pos = compute_adaptability(pos_emo_mean, pos_feat_mean)
        adaptability_neu = compute_adaptability(neu_emo_mean, neu_feat_mean)
        adaptability_neg = compute_adaptability(neg_emo_mean, neg_feat_mean)
        adaptability = (adaptability_pos + adaptability_neu + adaptability_neg) / 3
        
        # 计算一致性损失
        labels = labels.float()
        consistency_loss = self.alpha * variance_depression * labels + \
                         self.beta * (1 - adaptability) * (1 - labels)
        
        return consistency_loss.mean()

class EmotionEmbedding(nn.Module):
    def __init__(self, hidden_size=768, emotion_dim=768): 
        super().__init__()
        self.positive_fc = nn.Linear(hidden_size, emotion_dim) 
        self.neutral_fc = nn.Linear(hidden_size, emotion_dim)
        self.negative_fc = nn.Linear(hidden_size, emotion_dim)

    def forward(self, positive_features, neutral_features, negative_features):
        # 生成情绪嵌入
        positive_embeddings = self.positive_fc(positive_features)  # [batch_size, num_positive, emotion_dim]
        neutral_embeddings = self.neutral_fc(neutral_features)    # [batch_size, num_neutral, emotion_dim]
        negative_embeddings = self.negative_fc(negative_features)  # [batch_size, num_negative, emotion_dim]

        return positive_embeddings, neutral_embeddings, negative_embeddings
    
class StimulusEncoder(nn.Module):
    def __init__(self, dataset_name, num_stimuls, pretrained_model=config.ptm, hidden_size=768):
        super().__init__()
        self.dataset_name = dataset_name

        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.type_embeddings = nn.Embedding(3, hidden_size)  # 3种刺激类型
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        self.position_embeddings = nn.Embedding(num_stimuls, hidden_size)

        
    def forward(self, input_ids, attention_mask, stimulus_types, positions):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids[0], attention_mask=attention_mask[0])
        bert_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 获取类型和位置嵌入
        type_embeddings = self.type_embeddings(stimulus_types[0])  # [batch_size, hidden_size]
        pos_embeddings = self.position_embeddings(positions[0])  # [batch_size, hidden_size]
        
        # 融合所有特征
        combined = torch.cat([bert_embeddings, type_embeddings + pos_embeddings], dim=-1)
        stimulus_features = self.fusion_layer(combined)
        
        return stimulus_features

class ResponseEncoder(nn.Module):
    def __init__(self, pretrained_model=config.ptm, hidden_size=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

class InteractionLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.norm = nn.LayerNorm(hidden_size)

        
    def forward(self, stimulus_features, response_features):
        # stimulus_features: [num_stimulus, batch_size, hidden_size]
        # response_features: [num_stimulus, batch_size, hidden_size]
        
        attended_features, weights = self.attention(
            query=response_features,
            key=stimulus_features,
            value=stimulus_features
        )
        
        # 残差连接和归一化
        output = self.norm(attended_features + response_features)
        return output, weights
    

class RelationshipModule(nn.Module):
    def __init__(self, hidden_size=768, num_heads=2, dim_feedforward=768, dropout=0.5):
        super(RelationshipModule, self).__init__()

        # Transformer layers for different stimulus types
        self.positive_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )
        self.neutral_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )
        self.negative_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )
        
        self.emo_moel = EmotionEmbedding()

        self.pos_pos_gate = Local_GatedFusion(hidden_size=768)
        self.neu_neu_gate = Local_GatedFusion(hidden_size=768)
        self.neg_neg_gate = Local_GatedFusion(hidden_size=768)

        # Transformer layers for cross-stimulus interactions
        self.pos_neg_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )
        self.pos_neu_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )
        self.neg_neu_interaction = TransformerEncoder(
            TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, dropout),
            num_layers=1
        )

        self.features_reduce_pos = nn.Linear(3 * hidden_size, hidden_size)
        self.features_reduce_neu = nn.Linear(3 * hidden_size, hidden_size)
        self.features_reduce_neg = nn.Linear(3 * hidden_size, hidden_size)

        self.global_gate_fusion = Global_GatedFusion(hidden_size=768)

        # Final projection layer
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, interaction_features, stimulus_types, stimulus_emb):
        # interaction_features: [num_stimulus, batch_size, hidden_size]
        # stimulus_types: [batch_size, num_stimulus]
            
        # print('interaction_features:', interaction_features.shape)
        # print('stimulus_types:', stimulus_types.shape)

        batch_size = stimulus_types.size(0)
        num_stimulus = stimulus_types.size(1)
        # print('batch_size:', batch_size, 'num_stimulus:', num_stimulus)
        
        positive_features = []
        neutral_features = []
        negative_features = []

        positive_stimulus = []
        neutral_stimulus = []
        negative_stimulus = []

        for batch_idx in range(stimulus_types.size(0)):  # 遍历 batch_size
            # Mask for the current individual
            positive_mask = stimulus_types[batch_idx] == 1  # [num_stimulus]
            neutral_mask = stimulus_types[batch_idx] == 0  # [num_stimulus]
            negative_mask = stimulus_types[batch_idx] == 2  # [num_stimulus]
            
            # print('positive_mask:', positive_mask.shape, positive_mask)
            # print('neutral_mask:', neutral_mask.shape, neutral_mask)
            # print('negative_mask:', negative_mask.shape, negative_mask)

            # Extract features for the current individual
            positive_features.append(interaction_features[positive_mask, batch_idx, :])  # [num_positive, hidden_size]
            neutral_features.append(interaction_features[neutral_mask, batch_idx, :])    # [num_neutral, hidden_size]
            negative_features.append(interaction_features[negative_mask, batch_idx, :])  # [num_negative, hidden_size]

            positive_stimulus.append(stimulus_emb[positive_mask, batch_idx, :])  # [num_positive, hidden_size]
            neutral_stimulus.append(stimulus_emb[neutral_mask, batch_idx, :])    # [num_neutral, hidden_size]
            negative_stimulus.append(stimulus_emb[negative_mask, batch_idx, :])  # [num_negative, hidden_size]

        
        # Pad each individual's features to ensure equal length across the batch
        positive_features = nn.utils.rnn.pad_sequence(positive_features, batch_first=True)  # [batch_size, max_num_positive, hidden_size]
        neutral_features = nn.utils.rnn.pad_sequence(neutral_features, batch_first=True)    # [batch_size, max_num_neutral, hidden_size]
        negative_features = nn.utils.rnn.pad_sequence(negative_features, batch_first=True)  # [batch_size, max_num_negative, hidden_size]

        positive_stimulus = nn.utils.rnn.pad_sequence(positive_stimulus, batch_first=True)  # [batch_size, max_num_positive, hidden_size]
        neutral_stimulus = nn.utils.rnn.pad_sequence(neutral_stimulus, batch_first=True)    # [batch_size, max_num_neutral, hidden_size]
        negative_stimulus = nn.utils.rnn.pad_sequence(negative_stimulus, batch_first=True)  # [batch_size, max_num_negative, hidden_size]
        
        # print('positive_features:', positive_features.shape, 'positive_stimulus:', positive_stimulus.shape)
        # print('neutral_features:', neutral_features.shape, 'neutral_stimulus:', neutral_stimulus.shape)
        # print('negative_features:', negative_features.shape, 'negative_stimulus:', negative_stimulus.shape)


        # 刺激信息的隐含情绪嵌入生成
        positive_emo, neutral_emo, negative_emo = self.emo_moel(positive_stimulus, neutral_stimulus, negative_stimulus)
        # print('positive_emo:', positive_emo.shape)
        # print('neutral_emo:', neutral_emo.shape)
        # print('negative_emo:', negative_emo.shape)

        # # Internal interactions 
        pos_interacted = self.positive_interaction(positive_features.transpose(0, 1))  # Transformer expects [seq_len, batch_size, hidden_size]
        neu_interacted = self.neutral_interaction(neutral_features.transpose(0, 1))
        neg_interacted = self.negative_interaction(negative_features.transpose(0, 1))
        
        # # Local gate
        pos_interacted = self.pos_pos_gate(pos_interacted) # [max_num_positive, batch_size, hidden_size]
        neu_interacted = self.neu_neu_gate(neu_interacted) # [max_num_neutral, batch_size, hidden_size] 
        neg_interacted = self.neg_neg_gate(neg_interacted) # [max_num_negative, batch_size, hidden_size]

        # pos_interacted = positive_features.transpose(0, 1)  
        # neu_interacted = neutral_features.transpose(0, 1)
        # neg_interacted = negative_features.transpose(0, 1)

        # print('pos_interacted:', pos_interacted.shape) 
        # print('neu_interacted:', neu_interacted.shape)
        # print('neg_interacted:', neg_interacted.shape)

        # Extract global features for each stimulus type using multiple pooling methods
        pos_mean = pos_interacted.mean(dim=0, keepdim=True)  # Mean pooling
        pos_max = pos_interacted.max(dim=0, keepdim=True)[0]  # Max pooling
        pos_min = pos_interacted.min(dim=0, keepdim=True)[0]  # Min pooling
        pos_global = torch.cat([pos_mean, pos_max, pos_min], dim=-1) # [1, batch_size, 3*hidden_size]   

        neu_mean = neu_interacted.mean(dim=0, keepdim=True)
        neu_max = neu_interacted.max(dim=0, keepdim=True)[0]
        neu_min = neu_interacted.min(dim=0, keepdim=True)[0]
        neu_global = torch.cat([neu_mean, neu_max, neu_min], dim=-1) # [1, batch_size, 3*hidden_size]

        neg_mean = neg_interacted.mean(dim=0, keepdim=True)
        neg_max = neg_interacted.max(dim=0, keepdim=True)[0]
        neg_min = neg_interacted.min(dim=0, keepdim=True)[0]
        neg_global = torch.cat([neg_mean, neg_max, neg_min], dim=-1) # [1, batch_size, 3*hidden_size]
        
        # print('pos_global:', pos_global.shape)
        # print('neu_global:', neu_global.shape)
        # print('neg_global:', neg_global.shape)

        pos_global = self.features_reduce_pos(pos_global) # [1, batch_size, hidden_size]
        neu_global = self.features_reduce_pos(neu_global) # [1, batch_size, hidden_size]
        neg_global = self.features_reduce_pos(neg_global) # [1, batch_size, hidden_size]

        # print('pos_global_reduce:', pos_global.shape)
        # print('neu_global_reduce:', neu_global.shape)
        # print('neg_global_reduce:', neg_global.shape)
        
        # Cross-stimulus interaction
        pos_neg_features = torch.cat((pos_global, neg_global), dim=0) # [2, batch_size, hidden_size]
        pos_neu_features = torch.cat((pos_global, neu_global), dim=0) # [2, batch_size, hidden_size]
        neg_neu_features = torch.cat((neg_global, neu_global), dim=0) # [2, batch_size, hidden_size]

        pos_neg_interacted = self.pos_neg_interaction(pos_neg_features) # [2, batch_size, hidden_size]
        pos_neu_interacted = self.pos_neu_interaction(pos_neu_features) # [2, batch_size, hidden_size]
        neg_neu_interacted = self.neg_neu_interaction(neg_neu_features) # [2, batch_size, hidden_size]
        # print('pos_neg_interacted:', pos_neg_interacted.shape)
        # print('pos_neu_interacted:', pos_neu_interacted.shape)
        # print('neg_neu_interacted:', neg_neu_interacted.shape)

        all_final_out = self.global_gate_fusion(pos_neg_interacted, pos_neu_interacted, neg_neu_interacted)
        # print('all_final_out:', all_final_out.shape)

        return all_final_out, positive_emo, neutral_emo, negative_emo, pos_interacted.transpose(0,1), neu_interacted.transpose(0,1), neg_interacted.transpose(0,1)


class DepressionDetectionModel(nn.Module):
    def __init__(self, dataset_name, device, hidden_size=768, dropout=0.5):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = device

        if self.dataset_name == 'midd':
            self.num_stimulus = 9
        elif self.dataset_name == 'modma':
            self.num_stimulus = 18
        elif self.dataset_name.startswith("midd["):
            self.num_stimulus = 3
        elif self.dataset_name.startswith("modma["):
            self.num_stimulus = 6
        elif self.dataset_name == 'momda_to_midd' or 'midd_to_modma':
            self.num_stimulus = 9
        else:
            raise NotImplementedError

        print('num_stimuls: ', self.num_stimulus)
        
        # 编码器
        self.stimulus_encoder = StimulusEncoder(dataset_name=self.dataset_name, num_stimuls=self.num_stimulus)
        self.response_encoder = ResponseEncoder()
        
        # 交互层
        self.interaction = InteractionLayer()
        
        # 关系建模
        self.relationship = RelationshipModule(hidden_size=hidden_size, dropout=dropout)
        
        # 输出层
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size * self.num_stimulus, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 2)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, stimulus_data, response_data):
        """
        Args:
            stimulus_data: 包含input_ids, attention_mask, types, positions的字典
            response_data: 包含input_ids, attention_mask的字典
        """
        batch_size = response_data['input_ids'].size(0)
        # print(stimulus_data['input_ids'].shape, stimulus_data['attention_mask'].shape)
        # print(stimulus_data['types'].shape, stimulus_data['positions'].shape)
        # print(stimulus_data['types'])
        # print(stimulus_data['positions'])
        # 1. 编码刺激（只需计算一次）
        stimulus_features = self.stimulus_encoder(
            stimulus_data['input_ids'],
            stimulus_data['attention_mask'],
            stimulus_data['types'],
            stimulus_data['positions']
        )  
        
        # 扩展刺激特征以匹配batch_size
        stimulus_features = stimulus_features.unsqueeze(1).expand(-1, batch_size, -1)
        
        # 2. 编码每个被试的响应
        response_features = []
        for i in range(self.num_stimulus):
            # print(self.num_stimulus, i)
            response_i = self.response_encoder(
                response_data['input_ids'][:, i],
                response_data['attention_mask'][:, i]
            )  # [batch_size, hidden_size]
            response_features.append(response_i)
        response_features = torch.stack(response_features)  # [num_stimulus, batch_size, hidden_size]
        
        # print('response_features:', response_features.shape, 'stimulus_features:',stimulus_features.shape)
    
        # 3. 特征交互
        interaction_features, attention_weights = self.interaction(
            stimulus_features, response_features
        )  # [num_stimulus, batch_size, hidden_size]
        
        # print('interaction_features', interaction_features.shape)

        # 4. 关系建模
        relationship_features, positive_emo, neutral_emo, negative_emo, pos_interacted, neu_interacted, neg_interacted = self.relationship(interaction_features, response_data['stimuls_type'], stimulus_features)
        
        # print('relationship_features:', relationship_features.shape)
        # 5. 特征整合与分类
        features = relationship_features.transpose(0, 1)  # [batch_size, 2, hidden_size]  
        features = features.reshape(batch_size, -1)

        # print('features:', features.shape)
        # 输出预测
        output = self.classifier(features) # [batch_size, 2]
        

        output_log_prob = F.log_softmax(output, 1)  # 对数概率是为了MaskedKLDivLoss 准备的 KL损失接受的输入需要转换为log概率。
        
        return output, features, attention_weights, positive_emo, neutral_emo, negative_emo, pos_interacted, neu_interacted, neg_interacted


class bert_basline_model(nn.Module):
    def __init__(self, dataset_name, device, hidden_size=768, dropout=0.5):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = device

        if self.dataset_name == 'midd':
            self.num_stimulus = 9
        elif self.dataset_name == 'modma':
            self.num_stimulus = 18
        elif self.dataset_name in config.name_stimuls_3:
            self.num_stimulus = 3
        else:
            self.num_stimulus = 9

        print('num_stimuls: ', self.num_stimulus)
        
        # 编码器
        # self.stimulus_encoder = StimulusEncoder(dataset_name=self.dataset_name)
        self.response_encoder = ResponseEncoder()
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_stimulus, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, stimulus_data, response_data):
        batch_size = response_data['input_ids'].size(0)
        
        # 编码每个被试的响应
        response_features = []
        for i in range(self.num_stimulus):
            # print(self.num_stimulus, i)
            response_i = self.response_encoder(
                response_data['input_ids'][:, i],
                response_data['attention_mask'][:, i]
            )  # [batch_size, hidden_size]
            response_features.append(response_i)
        response_features = torch.stack(response_features)  # [num_stimulus, batch_size, hidden_size]
        
        
        features = response_features.transpose(0, 1)  # [batch_size, num_stimulus, hidden_size]  
        features = features.reshape(batch_size, -1)

        # print('features:', features.shape)
        # 输出预测
        output = self.classifier(features) # [batch_size, 2]
        
        output_log_prob = None
        attention_weights = None
        return output, output_log_prob, attention_weights



def create_model(dataset_name, device, hidden_size=768, dropout=0.5):
    print('Model: DepressionDetectionModel')
    model = DepressionDetectionModel(dataset_name=dataset_name, device=device, hidden_size=hidden_size, dropout=dropout)
    return model


def create_bert_basline_model(dataset_name, device, hidden_size=768, dropout=0.5):
    model = bert_basline_model(dataset_name=dataset_name, device=device, hidden_size=hidden_size, dropout=dropout)
    return model