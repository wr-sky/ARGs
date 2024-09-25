import torch
import esm
import pickle
from tqdm.auto import tqdm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter(1022)
device_1 = torch.device("cpu")
device_2 = torch.device("cuda")
model.to(device_2)  # 将模型移动到 GPU 设备上
#
model.eval()  # disables dropout for deterministic results

# data_path = "/data/newdisk/mrj_data/resistance/deeparg_data/deepARG_NS_data.pkl"
data_path = "/data/newdisk/mrj_data/resistance/hdm_data/HDM_NS_data.pkl"

with open(data_path, 'rb') as f:
    data_1 = pickle.load(f)

print(data_1[0])
data = []
for i in range(len((data_1))):
    print(data_1[i])
    data_tuple = ()
    data_label = list(data_1[i].keys())
    data_seq = data_1[i][data_label[0]]
    # data_seq = data_seq
    data_2 = data_tuple + (data_label[0],data_seq)
    data.append(data_2)

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

batch_tokens = batch_tokens.to(device_2)
# Extract per-residue representations
results_data = []
for i in tqdm(range(len(batch_tokens))):
#
    with torch.no_grad():
        results = model(batch_tokens[i].unsqueeze(0), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        # print(token_representations.shape)
        token_representations = token_representations.cpu()
        results_data.append(token_representations)


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

name = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_ESM_emb"
# name = "/data/newdisk/mrj_data/resistance/hdm_data/exp_data/HDM_NS_ESM_emb"

save_dict(results_data,name)