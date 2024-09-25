from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from tqdm.auto import tqdm
import torch


data_path = "/data/newdisk/mrj_data/resistance/card_prot_perfect_strict_emb.pkl"
label_path = "/data/newdisk/mrj_data/resistance/CARD_id.txt"

with open(data_path, 'rb') as f:
    strain_data = pickle.load(f)
# print(strain_data[0])

def  strain_label(strain_data,label_path):
    bacteria_db = []
    for i in tqdm(range(len(strain_data))):
        bacteria_data = []
        # print("strain_data:",strain_data)
        for j in tqdm(range(len(strain_data[i]))):
            protein_sequence_1 = strain_data[i][j]["emb"]
            protein_sequence_1 = protein_sequence_1.view(-1)

            emb_len = len(protein_sequence_1)
            padding_len = 30720 - emb_len
            emb_vector = torch.cat((protein_sequence_1.cuda(), torch.zeros(padding_len).cuda()))

            emb_vector = emb_vector.cpu()

            vector_np = emb_vector.numpy().reshape(-1, 1)
            scaler = StandardScaler()
            scaled_vector = scaler.fit_transform(vector_np)
            scaled_emb = torch.Tensor(scaled_vector)

            protein_id = strain_data[i][0]["label"]["labels"]
            label_ids, strain_labels_list = label_id(label_path)
            if protein_id[0] in label_ids:
                strain_id = label_ids.index(protein_id[0])

                strain_label = strain_labels_list[strain_id]

                protein_dict = {"emb":scaled_emb,"label":strain_label}
                # print(protein_dict)
                bacteria_data.append(protein_dict)

        bacteria_db.append(bacteria_data)

    return bacteria_db


def label_id(label_path):
    # labels_list = ["macrolide-lincosamide-streptogramin", "multidrug", "others", "tetracycline", "quinolone",
    #                "aminoglycoside", "bacitracin", "beta_lactam", "fosfomycin",
    #                "glycopeptide", "chloramphenicol", "rifampin", "sulfonamide", "trimethoprim", "polymyxin"]

    labels_list = ['multidrug','beta_lactam','aminoglycoside','rifampin','others','tetracycline','quinolone','macrolide-lincosamide-streptogramin','fosfomycin',
                   'polymyxin','chloramphenicol','bacitracin','trimethoprim','sulfonamide','glycopeptide','non']

    # 获取标签文档
    strain_labels_list = []
    label_id = []
    with open(label_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        id = lines[i].strip(" ").split('\t')[0]
        label_id.append(id)
        resistance_label = lines[i].strip(" ").split('\t')[2:]
        resistance_label = [label.rstrip("\n") for label in resistance_label]
        strain_label_list = []
        for label_1 in resistance_label:
            if label_1 in labels_list:
                label = labels_list.index(label_1)
                strain_label_list.append(label)
        strain_labels_list.append(strain_label_list)

    return label_id,strain_labels_list

database = strain_label(strain_data,label_path)

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

name = "/data/newdisk/mrj_data/resistance/card_prot_perfect_strict_emb_data"

save_dict(database,name)