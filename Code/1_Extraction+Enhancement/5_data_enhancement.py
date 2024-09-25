import torch
import pickle
from tqdm.auto import tqdm
import numpy as np
from sklearn.decomposition import PCA

# data_path = "/data/newdisk/mrj_data/resistance/hdm_data/exp_data/HDM_prottrans.pkl"
# data_path_1 = "/data/newdisk/mrj_data/resistance/hdm_data/exp_data/HDM_ESM.pkl"
# data_path = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_prottrans.pkl"
# data_path_1 = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_ESM.pkl"
data_path = "/data/newdisk/mrj_data/resistance/hdm_data/exp_data/train_HDM_fusion.pkl"

with open(data_path, 'rb') as f:
    data = pickle.load(f)

def feature_fusion (data):

    label_list = [3,4,8,9,10,11,12,13]
    fusion_data = []
    for i in tqdm(range(len(data))):

        emb_label = data[i]["label"]

        if emb_label in label_list:
            emb_1 = data[i]["emb_1"]
            emb_2 = data[i]["emb_2"]
            emb_2 = emb_2.flatten()
            reshaped_tensors = emb_2.reshape(-1, 1280)
            data_pca = []
            for i in range(len(reshaped_tensors)):
                reshaped_tensor = reshaped_tensors[i]
                # print("reshaped_tensor:",len(reshaped_tensor))
                reshaped_tensor = reshaped_tensor.reshape(32, -1)
                pca = PCA(n_components=1)
                emb_vector_1 = pca.fit_transform(reshaped_tensor)
                # print("pac:",len(emb_vector_1))
                emb_vector_1_tensor = torch.tensor(emb_vector_1)
                data_pca.append(emb_vector_1_tensor)

            emb_vectors_1 = torch.cat(data_pca, dim=0)
            emb_vectors_1 = emb_vectors_1[:30720]

            emb_vectors_2 = np.tile(emb_1, 43).reshape(-1)
            # emb_vectors_3 = torch.from_numpy(emb_vectors_2)
            emb_vectors_4 = emb_vectors_2[:1310720]

            emb_vectors_4_tensor = torch.tensor(emb_vectors_4)

            enhancement_dict = {"emb_1": emb_vectors_1,"emb_2": emb_vectors_4_tensor, "label": emb_label}
            fusion_data.append(enhancement_dict)
        else:
            continue

    return fusion_data

fusion_data = feature_fusion(data)

for i in tqdm(range(len(fusion_data))):
    data.append(fusion_data[i])


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_dict(data,"/data/newdisk/mrj_data/resistance/hdm_data/exp_data/train_HDM_fusion_enhancement")

# with open(data_path_1, 'rb') as f:
#     data_1 = pickle.load(f)



# def feature_fusion (data,data_1):
#
#     fusion_data = []
#     for i in tqdm(range(len(data))):
#         emb_label = data[i]["label"]
#         emb_1 = data[i]["emb"]
#         emb_1 = emb_1.view(-1)
#         # print(emb_1)
#         emb_1_len = len(emb_1)
#         padding_len = 30720 - emb_1_len
#         zero_tensor = torch.zeros(padding_len).cuda()
#         # print(zero_tensor)
#         emb_vectors = torch.cat((emb_1,zero_tensor))
#         emb_vectors = emb_vectors.cpu()
#         emb_vectors = np.tile(emb_vectors, 43)
#         emb_vectors = torch.from_numpy(emb_vectors)
#
#         emb_vectors = emb_vectors[:1310720]
#         fusion_dict_1 = {"emb": emb_vectors, "label": emb_label}
#         fusion_data.append(fusion_dict_1)
#         # print(emb_vectors)
#         print("len:",len(emb_vectors))
#
#         emb_2 = data_1[i]
#
#         emb_2 = emb_2.view(-1)
#
#         fusion_dict_2 = {"emb":emb_2,"label":emb_label}
#         fusion_data.append(fusion_dict_2)
#
#     return fusion_data
#
# fusion_data = feature_fusion(data,data_1)
# print(len(fusion_data))
#
# def save_dict(obj, name):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#
# save_dict(fusion_data,"/data/newdisk/mrj_data/resistance/fusion_data/deepARG_data_enhancement")