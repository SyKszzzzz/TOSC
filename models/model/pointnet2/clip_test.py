from models.model.clip_sd import ClipCustom
import torch
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

def pairwise_cosine_sim(text_feats: torch.Tensor):
    """
    输入: 
      text_feats (N, D) 未归一化的特征
    输出:
      sim_matrix (N, N) 余弦相似度矩阵
    """
    # 1. 先归一化
    text_feats = F.normalize(text_feats, dim=1)  # (N, D), 每行单位范数

    # 2. 计算两两相似度
    sim_matrix = text_feats @ text_feats.t()     # (N, N)
    return sim_matrix


def pairwise_cosine_similarity(feats: torch.Tensor):
    # feats: (B, D)，且已归一化。相乘即余弦相似度。
    return feats @ feats.t()  # (B, B)

def extract_sentence_features(cond_txt: torch.Tensor,
                              token_ids: torch.Tensor,
                              eot_token_id: int = 49407):
    """
    从 per-token 特征 (B, L, D) 里，取出每句的句向量 (B, D)。
    CLIP 做法是：在每条句子里的 <EOS> token 处取 output。
    """
    B, L, D = cond_txt.shape
    # 找到每个序列 EOS 的位置；如果有多个 EOS，argmax 会拿到第一个
    # 注意 token_ids 里 EOS 的 id 在不同版本 CLIP 里可能不一样
    eot_positions = (token_ids == eot_token_id).int().argmax(dim=1)  # (B,)
    # gather 出每条句子的 EOS 处的隐藏向量
    batch_idx = torch.arange(B, device=cond_txt.device)
    sent_feats = cond_txt[batch_idx, eot_positions]  # (B, D)
    # L2 归一化
    sent_feats = F.normalize(sent_feats, dim=-1)
    return sent_feats  # float32 (B, D)

def threshold_clustering(sim_mat: torch.Tensor, threshold: float = 0.9):
    """
    用简单的连通分量方法，把 sim >= threshold 的节点连在一起当一类。
    返回每个样本的 cluster id (0,1,2...)
    """
    N = sim_mat.size(0)
    visited = [False]*N
    clusters = [-1]*N
    cid = 0
    adj = (sim_mat >= threshold).cpu().numpy()

    def dfs(u):
        stack = [u]
        while stack:
            v = stack.pop()
            for w in range(N):
                if adj[v, w] and not visited[w]:
                    visited[w] = True
                    clusters[w] = cid
                    stack.append(w)

    for i in range(N):
        if not visited[i]:
            visited[i] = True
            clusters[i] = cid
            dfs(i)
            cid += 1

    return clusters

if __name__ == "__main__":
    # 假设 text_vector 已经在 GPU 上
    language_model = ClipCustom(cfg=None,num=384).to("cuda")
    language_model.freeze()
    # guidance = ["Could you please lightly touch the lens", "The lens of the binoculars is gripped to replace or remove lens caps.", "The binoculars' lens is being softly grasped.","Gently hold the binoculars.", "Contact the binoculars' lens to replace or remove lens caps.", "The binoculars are being gently contacted to hold them steady for clear viewing.", "How should I hold binoculars steady for clear viewing using three fingers?"]

    # guidance = ["Could you please lightly touch the lens", "The lens of the binoculars is gripped to replace or remove lens caps.", "The binoculars' lens is being softly grasped.","Gently hold the binoculars.", "Contact the binoculars' lens to replace or remove lens caps.", "The binoculars are being gently contacted to hold them steady for clear viewing.", "How should I hold binoculars steady for clear viewing using three fingers?"]


    # guidance = ["Could you please lightly touch the lens", "The lens of the binoculars is gripped to replace or remove lens caps.", "The binoculars' lens is being softly grasped.","Gently hold the binoculars.", "Contact the binoculars' lens to replace or remove lens caps.", "The binoculars are being gently contacted to hold them steady for clear viewing.", "How should I hold binoculars steady for clear viewing using three fingers?"]

    guidance = ["How do I hold the handle of a mug with my index, middle, and thumb fingers?", "How should I grasp the mug's handle to drink hot beverages like coffee or tea using three fingers?", "How can I grip the mug's handle to hang it on a rack?", "How should I firmly grasp the handle using four fingers to maintain a slip-free grip, especially with wet or slick hands?", "Can you describe how to hold a mug firmly with four fingers?"]
    
    cond_txt, text_vector = language_model(guidance, None)
    eot_id = language_model.model.token_embedding.num_embeddings - 1  # 或者直接 49407
    sent_feats = extract_sentence_features(cond_txt, text_vector, eot_token_id=eot_id)
    sim_mat = pairwise_cosine_similarity(sent_feats).cpu().detach().numpy()
    print("Cosine sim mat:\n", np.round(sim_mat, 3))

    # 3) 用 seaborn 画一下相似度 heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(sim_mat, annot=True, cmap="vlag", vmin=0, vmax=1,
                xticklabels=guidance, yticklabels=guidance)
    plt.title("Text cosine similarity")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 4) 用 t-SNE / PCA 可视化到 2D
    X = sent_feats.detach().cpu().numpy()
    # 4a) t-SNE
    tsne = TSNE(n_components=2, metric="cosine", init="random", random_state=42)
    X2d = tsne.fit_transform(X)
    # 4b) 如果想试 PCA，只需换成下面一行：
    # X2d = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(6,6))
    plt.scatter(X2d[:,0], X2d[:,1], c="C0")
    for i, txt in enumerate(guidance):
        plt.annotate(f"{i}", (X2d[i,0], X2d[i,1]))
    plt.title("t-SNE of text features")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()


    # print("text_vector shape:", text_vector.shape, cond_txt)  # Tensor (N, D) torch.Size([6, 77])
    # print("cond_txt shape:", cond_txt.shape)  # Tensor (N, D) torch.Size([6, 384])
    # sim = pairwise_cosine_sim(text_vector.float())  # Tensor (N, N)

    # # 打印相似度矩阵
    # sim_np = sim.cpu().detach().numpy()
    # import numpy as np
    # np.set_printoptions(precision=3, suppress=True)
    # print("Cosine Similarity:\n", sim_np)

    # # 用阈值聚类
    # threshold = 0.95
    # labels = threshold_clustering(sim, threshold)
    # print("Cluster labels (threshold=%.2f): %s" % (threshold, labels))