import torch
from tqdm.auto import tqdm
import pickle
from sklearn.preprocessing import StandardScaler

data_path = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_prottrans_emb.pkl"
data_path_1 = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_ESM_emb.pkl"

with open(data_path, 'rb') as f:
    data = pickle.load(f)
print(data[0])

with open(data_path_1, 'rb') as f:
    data_1 = pickle.load(f)

print(len(data_1))
print(data_1[0])
print(data_1[1].shape)

# 给ESM提取数据添加标签
def feature_fusion (data,data_1):

    fusion_data = []
    for i in tqdm(range(len(data_1))):
        emb_label = data[i]["label"]
        emb = data_1[i]
        emb = emb.view(-1)
        vector = emb.cpu()
        vector_np = vector.numpy().reshape(-1, 1)
        scaler = StandardScaler()
        scaled_vector = scaler.fit_transform(vector_np)
        scaled_emb = torch.Tensor(scaled_vector)
        fusion_dict = {"emb": scaled_emb,"label":emb_label}
        fusion_data.append(fusion_dict)

    return fusion_data


fusion_data = feature_fusion(data,data_1)

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_dict(fusion_data,"/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_ESM_uniform")