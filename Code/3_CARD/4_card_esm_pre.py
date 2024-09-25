import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.utils import collate_tensors
import pytorch_lightning as pl
from pytorch_lightning.metrics.sklearns import Accuracy,F1
import pickle
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
from torchnlp.datasets.dataset import Dataset
from pytorch_lightning import Trainer
from collections import defaultdict
import math
import tqdm

# ckpt_path = "/home/mrj1990/e_codes/strain_lstmexperiments/lightning_logs/version_25-03-2024--20-20-08/checkpoints/epoch=145-val_loss=0.35-val_acc=0.95.ckpt"
# data_path = "/data/newdisk/mrj_data/resistance/strain_database.pkl"
ckpt_path = "/home/mrj1990/e_codes/ESMexperiments/lightning_logs/version_07-04-2024--08-43-28/checkpoints/epoch=112-val_loss=0.22-val_acc=0.97.ckpt"
class LSTMFlatten(nn.Module):
    def forward(self, x):
        output, _ = x
        batch_size = output.shape[0]
        flattened_output = output.reshape(batch_size, -1)
        return flattened_output

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiheadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        # print(query.shape)
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1,2)  # (batch_size, num_heads, seq_len, head_size)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1,2)  # (batch_size, num_heads, seq_len, head_size)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1,2)  # (batch_size, num_heads, seq_len, head_size)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attention_scores.masked_fill_(mask == 0, -1e9)  # Set scores to -infinity for masked positions
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        context_vector = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, -1,self.hidden_size)  # (batch_size, seq_len, hidden_size)

        # Output linear transformation
        # output = self.fc(context_vector)
        return context_vector, attention_weights


class LSTMWithMultiheadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(LSTMWithMultiheadAttention, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.256)
        # self.dropout = nn.Dropout(p=0.256)
        self.attention = MultiheadAttention(hidden_size, num_heads)

    def forward(self, input):
        encoder_outputs, (hidden, _) = self.encoder(input)

        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs, encoder_outputs)

        return context_vector, attention_weights

class Strain_pre(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super(Strain_pre, self).__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size
        self.metric_acc = Accuracy()
        self.__build_model()
        self.pre_label = []

    def __build_model(self) -> None:

        self.hidden_size_1 = 512
        self.hidden_size_2 = 1024
        self.hidden_size_3 = 2048
        # self.hidden_size_4 = 4096

        self.classification_head = nn.Sequential(

            LSTMWithMultiheadAttention(input_size=1280, hidden_size=512, num_layers=6, num_heads=4),
            LSTMFlatten(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.Dropout(p=0.256),
            nn.GELU(),

            nn.InstanceNorm1d(num_features=self.hidden_size_2),
            nn.Linear(self.hidden_size_2, self.hidden_size_3),
            nn.Dropout(p=0.365),
            nn.GELU(),

            nn.InstanceNorm1d(num_features=self.hidden_size_3),
            nn.Linear(self.hidden_size_3, 16),
        )

    def forward(self, input):
        return {"logits": self.classification_head(input)}

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        targets_label = []
        inputs, targets = batch
        # print(targets)
        model_out = self.forward(inputs)

        y_hat = model_out["logits"]
        labels_hat = torch.argmax(y_hat, dim=1)

        for i in range(len(targets)):
            targets_label.append(int(targets[i][0]))

        count_correct = 0
        for target_label in targets_label:
            if target_label in labels_hat:
                count_correct += 1

        total_samples = len(targets_label)
        if total_samples != 0:
            percentage_correct = (count_correct / total_samples) * 100
        else:
            return None

        tqdm_dict = {"strain_label": labels_hat}
        output = OrderedDict(
            {
                "log": tqdm_dict,
                "strain_label": labels_hat,
                "target_label": targets_label,
                "percentage_correct": percentage_correct,
            }
        )
        return output

    def test_epoch_end(self, outputs: list) -> dict:

        self.label_counts = defaultdict(int)
        for x in outputs:
            target_label = x["target_label"]
            for value in x["strain_label"].tolist():
                if value != 15:
                    self.label_counts[value] += 1
        sorted_label_counts = sorted(self.label_counts.items(), key=lambda item: item[1], reverse=True)

        total_percentage_correct = 0
        total_samples = 0
        for x in outputs:
            total_percentage_correct += x["percentage_correct"]
            total_samples += len(x["strain_label"])

        if len(outputs) != 0:
            average_percentage_correct = total_percentage_correct / len(outputs)
        else:
            return None

        result = {
            "target_label": target_label,
            "pre_label_counts": sorted_label_counts,
            "average_percentage_correct": average_percentage_correct,
        }
        return result

    def configure_optimizers(self):
        parameters = [{"params": self.classification_head.parameters()}]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:

        parser.opt_list(
             "--learning_rate",
            default=2e-04,
            type=float,
            help="Classification head learning rate.",
        )

        parser.add_argument(
            "--loader_workers",
            default=36,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )

        parser.add_argument(
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        return parser

parser = HyperOptArgumentParser(
    strategy="random_search",
    description="Minimalist pre Classifier",
    add_help=True,
)
parser.add_argument(
    "--precision", type=int, default="32", help="full precision or mixed precision mode"
)

parser.add_argument("--amp_level", type=str, default="O3", help="mixed precision type")
parser = Strain_pre.add_model_specific_args(parser)
hparams = parser.parse_known_args()[0]

model = Strain_pre.load_from_checkpoint(ckpt_path)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

trainer = Trainer(
    precision=hparams.precision,
    amp_level=hparams.amp_level,
    deterministic=True,
    gpus=-1,
)

def prepare_sample(sample):
    sample = collate_tensors(sample)
    inputs = sample["emb"]
    # print(sample)
    targets = torch.tensor(sample["label"])
    return inputs, targets

# path = "/data/newdisk/mrj_data/resistance/strain_database.pkl"
path = "/data/newdisk/mrj_data/resistance/esm_perfect_card_data_emb.pkl"
def load_dataset(path):

        with open(path, 'rb') as f:
            dataset_list = pickle.load(f)
        return dataset_list

data_list = load_dataset(path)

def load_pre(strain_list):

    data = []
    for i in range(len(strain_list)):
        label = strain_list[i]["label"]
        if len(label) == 0:
           return None
        vector = strain_list[i]["emb"]
        dict = {"emb": vector.view(-1, 1280), "label": label}
        # print(dict)
        data.append(dict)
    dataset_1 = Dataset(data)
    strain_data = DataLoader(
        dataset=dataset_1,
        batch_size=128,
        collate_fn=prepare_sample,
        num_workers=36,
    )
    with torch.no_grad():
        strain_output = trainer.test(model, strain_data)

    return strain_output

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

pre_results = []
targets_results = []
a =0
for i in range(len(data_list)):
    # print(len(data_list[28]))
    if len(data_list[i]) == 0:
        a = a +1
        print(a)
        continue
    else:
        ouputs = load_pre(data_list[i])
        if ouputs != None:
            pre_results.append(ouputs[0]["pre_label_counts"])
            targets_results.append(ouputs[0]["target_label"])
        else:
            continue

save_dict(pre_results, "/data/newdisk/mrj_data/resistance/pre_perfect_results")
save_dict(targets_results, "/data/newdisk/mrj_data/resistance/targets_perfect_results")


