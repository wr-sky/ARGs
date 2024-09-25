import torch
from tqdm.auto import tqdm
import pickle
from sklearn.preprocessing import StandardScaler


path = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_prottrans_emb.pkl"
path = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_prottrans_emb_pool.pkl"

with open(path, 'rb') as f:
    content = pickle.load(f)

print(content[10])
print(content[10]["emb"].shape)
print(content[0])
print(content[0]["emb"].shape)

def load_data(path):

    with open(path, 'rb') as f:
        content = pickle.load(f)

    data_uniform = []
    for i in range(len(content)):
        strain_label = content[i]["label"]
        emb = content[i]["emb"]
        emb = emb.view(-1)

        emb_len = len(emb)
        padding_len = 30720 - emb_len
        zero_tensor = torch.zeros(padding_len).cuda()
        # print(zero_tensor)
        emb_vector = torch.cat((emb,zero_tensor))
        emb_vector = emb_vector.cpu()

        vector_np = emb_vector.numpy().reshape(-1, 1)
        scaler = StandardScaler()
        scaled_vector = scaler.fit_transform(vector_np)

        scaled_emb = torch.Tensor(scaled_vector)

        data_dict = {"emb": scaled_emb, "label": strain_label}
        data_uniform.append(data_dict)

    return data_uniform

strain_data_1 = load_data(path)
print(strain_data_1)

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_dict(strain_data_1,"/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_prottrans_emb_uniform")

