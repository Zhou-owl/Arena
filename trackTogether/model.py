import torch
import torch.nn as nn
import torch.optim as optim
class PointNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(PointNet, self).__init__()

        # 嵌入网络
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # 聚类层
        self.cluster_layer = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        # 噪声点识别层
        self.noise_detection_layer = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0到1之间的概率值表示是否是噪声点
        )

    def forward(self, x):
        # 嵌入网络
        embedding = self.embedding_network(x)

        # 聚类层
        cluster_output = self.cluster_layer(embedding)

        # 噪声点识别层
        noise_prob = self.noise_detection_layer(embedding)

        return cluster_output, noise_prob
def compute_loss(cluster_output, noise_prob, lambda_inter, lambda_intra, lambda_embed, lambda_penalty):
    # 计算点云之间的距离损失，这里可以使用点云之间的最小距离作为衡量
    inter_loss = torch.mean(torch.sum((cluster_output.unsqueeze(1) - cluster_output.unsqueeze(0)) ** 2, dim=-1))

    # 计算点云内部的紧密程度损失，可以使用每个点云集内点的平均距离作为衡量
    intra_loss = torch.mean(torch.sum((cluster_output.unsqueeze(1) - cluster_output.unsqueeze(0)) ** 2, dim=-1))

    # 嵌入空间中的距离损失，可以使用点向量之间的距离作为衡量
    # 在这里我们可以使用欧氏距离
    pairwise_distances = torch.cdist(embedding, embedding, p=2)  # 计算所有点对之间的欧氏距离
    embed_loss = torch.mean(pairwise_distances)

    # 聚类数量的惩罚项，根据预期聚类数量和实际聚类数量之间的差异
    penalty_loss = torch.abs(torch.mean(torch.sum(cluster_output, dim=0)) - m_expected)

    return inter_loss, intra_loss, embed_loss, penalty_loss
# 定义超参数
input_dim = 4  # 输入点的维度（位置、物体类型、颜色信息、置信度）
embedding_dim = 64  # 嵌入维度
output_dim = 3  # 输出点的维度（聚类中心）

lambda_inter = 0.5
lambda_intra = 0.5
lambda_embed = 0.5
lambda_penalty = 0.1

# 实例化网络
net = PointNet(input_dim, embedding_dim, output_dim)

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 进行训练
for epoch in range(num_epochs):
    optimizer.zero_grad()
    cluster_output, noise_prob = net(input_points)
    inter_loss, intra_loss, embed_loss, penalty_loss = compute_loss(cluster_output, noise_prob, lambda_inter, lambda_intra, lambda_embed, lambda_penalty)
    total_loss = lambda_inter * inter_loss + lambda_intra * intra_loss + lambda_embed * embed_loss + lambda_penalty * penalty_loss
    total_loss.backward()
    optimizer.step()
