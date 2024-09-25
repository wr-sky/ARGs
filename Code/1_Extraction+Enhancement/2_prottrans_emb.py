import torch
import pickle
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

#加载预训练模型，并将模型放在GPU上
model_name = "Rostlab/prot_bert_bfd"
device = torch.device('cuda:1')

ProtBertBFD = BertForMaskedLM.from_pretrained(model_name)

ProtBertBFD = ProtBertBFD.cuda()

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

def Protbertemmbading(input_ids, token_type_ids, attention_mask,**kwds):

    for param in ProtBertBFD.parameters():
        param.requires_grad = False
    ProtBertBFD.eval()
    with torch.no_grad():
        input_ids = torch.tensor(input_ids).to(ProtBertBFD.device)
        attention_mask = torch.tensor(attention_mask).to(ProtBertBFD.device)

        embedding_data = ProtBertBFD(input_ids, attention_mask)[0]

    return embedding_data

#处理蛋白质文本数据，将其转换为预训练模型可以接受的输入格式
def prepare_sample(sample: list):
    # sample = collate_tensors(sample)  #根据需求对张量进行处理、组合和转换
    sample = collate_tensors(sample)

    inputs = tokenizer.batch_encode_plus(
        sample["seq"],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=1024,
    )  #将文本转换为模型可以以接受的输入格式

    return inputs


def emb_lists(seq: list, label: list):

    input_1 = [{"seq": seq, "label": label}]
    input_2 = Dataset(input_1)
    input = prepare_sample(input_2)
    input = input

    result = Protbertemmbading(input["input_ids"],input["token_type_ids"],input["attention_mask"])

    return result

#获取data中的每一个细菌菌株，并输入emb_lists进行处理
def embedding_dataset(path):

    with open(path, 'rb') as f:
        content = pickle.load(f)

    pro_eb_datasets = []
    for i in tqdm(range(len(content))):
        df = content[i]
        seq = list(df.values())[0]
        label = list(df.keys())[0]
        seq = " ".join("".join(seq.split()))

        pro_emb_dataset = emb_lists(seq,label)
        pro_emb_dict = {"emb": pro_emb_dataset, "label": label}
        pro_eb_datasets.append( pro_emb_dict)

    return pro_eb_datasets


path = "/data/newdisk/mrj_data/resistance/deeparg_data/deepARG_NS_data.pkl"
name = "/data/newdisk/mrj_data/resistance/deeparg_data/exp_data/deepARG_NS_prottrans_emb"

embedding_dataset = embedding_dataset(path)




def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


save_dict(embedding_dataset, name)
