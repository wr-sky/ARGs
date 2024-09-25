import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.utils import collate_tensors
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.sklearns import Accuracy,F1
import pickle
from test_tube import HyperOptArgumentParser
import os
from datetime import datetime
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from torchnlp.datasets.dataset import Dataset




class MS_dataset:

    def __init__(self) -> None:
        None


    def load_dataset(self, path):
        with open(path, 'rb') as f:
            self.strain_vector = pickle.load(f)

        self.strain_data = []

        for i in range(len(self.strain_vector)):
            self.strain_label = self.strain_vector[i]["label"]
            self.strain_vectors = self.strain_vector[i]["emb"]
            self.strain_vectors = self.strain_vectors.cpu()

            emb_np = self.strain_vectors.numpy().reshape(-1, 1)
            scaler = StandardScaler()
            scaled_emb = scaler.fit_transform(emb_np)

            scaled_emb = torch.Tensor(scaled_emb)

            self.strain_dict = {"emb": scaled_emb.view(-1, 512), "label": self.strain_label}
            self.strain_data.append(self.strain_dict)

        return Dataset(self.strain_data)

class LSTMFlatten(nn.Module):
    def forward(self, x):
        # x 是一个元组，包含了输出张量和最后一个时间步的隐藏状态
        output, _ = x
        batch_size = output.shape[0]
        flattened_output = output.reshape(batch_size, -1)
        return flattened_output

class ProtAlbertClassifier(pl.LightningModule):

    def __init__(self, hparams) -> None:
        super(ProtAlbertClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size

        # self.actual_labels_a_0 = 0
        # self.actual_labels_a_1 = 0
        # self.actual_labels_b_0 = 0
        # self.actual_labels_b_1 = 0

        self.dataset = MS_dataset()

        self.metric_acc = Accuracy()
        # self.metric_F1 = F1(average='macro')

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

    def __build_model(self) -> None:

        self.hidden_size_1 = 350720
        self.hidden_size_2 = 512
        self.hidden_size_3 = 1024

        self.classification_head = nn.Sequential(

            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.456),
            nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3),
            nn.Flatten(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.Dropout(p=0.456),
            nn.GELU(),

            nn.InstanceNorm1d(num_features=self.hidden_size_2),
            nn.Linear(self.hidden_size_2, self.hidden_size_3),
            nn.Dropout(p=0.365),
            nn.GELU(),

            nn.InstanceNorm1d(num_features=self.hidden_size_3),
            nn.Linear(self.hidden_size_3, 32),
            nn.Dropout(p=0.356),
            # nn.Softmax(dim=-1),
        )

    def __build_loss(self):

        self._loss = nn.CrossEntropyLoss()

    def forward(self, input):

        return {"logits": self.classification_head(input)}

    def loss(self, predictions: dict, targets:dict):

        return self._loss(predictions["logits"], targets)


    def prepare_sample(self, sample):

        sample = collate_tensors(sample)

        inputs = sample["emb"]
        targets = torch.tensor(sample["label"])

        return inputs, targets

    def training_step(self, batch, batch_nb, *args, **kwargs) -> dict:

        inputs, targets = batch

        model_out = self.forward( inputs)

        loss_val = self.loss(model_out, targets)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:

        inputs, targets = batch

        model_out = self.forward(inputs)

        loss_val = self.loss(model_out, targets)

        y = targets
        y_hat = model_out["logits"]

        labels_hat = torch.argmax(y_hat, dim=1)

        val_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict(
            {
                "val_loss": loss_val,
                "val_acc": val_acc,
            }
        )

        return output

    def validation_epoch_end(self, outputs: list) -> dict:

        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()

        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:

        inputs, targets = batch

        model_out = self.forward(inputs)
        loss_test = self.loss(model_out, targets)

        y = targets
        y_hat = model_out["logits"]

        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = self.metric_acc(labels_hat, y)
        # test_f1 = self.metric_F1(y, labels_hat)

        # if y == 0 and labels_hat == 0:
        #     self.actual_labels_a_0 = self.actual_labels_a_0 + 1
        # elif y == 0 and labels_hat == 1:
        #     self.actual_labels_a_1 = self.actual_labels_a_1 + 1
        # elif y == 1 and labels_hat == 0:
        #     self.actual_labels_b_0 = self.actual_labels_b_0 + 1
        # else:
        #     self.actual_labels_b_1 = self.actual_labels_b_1 + 1

        output = OrderedDict(
            {
                "test_loss": loss_test,
                "test_acc": test_acc,
                # "test_f1": test_f1,
                # "actual_labels_a_0": self.actual_labels_a_0,
                # "actual_labels_a_1": self.actual_labels_a_1,
                # "actual_labels_b_0": self.actual_labels_b_0,
                # "actual_labels_b_1": self.actual_labels_b_1,
            }
        )

        return output

    def test_epoch_end(self, outputs: list) -> dict:

        # print("actual_labels_a_0:", [x["actual_labels_a_0"] for x in outputs])
        # print("actual_labels_a_1", [x["actual_labels_a_1"] for x in outputs])
        # print("actual_labels_b_0", [x["actual_labels_b_0"] for x in outputs])
        # print("actual_labels_b_1", [x["actual_labels_b_1"] for x in outputs])

        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc_mean = torch.stack([x["test_acc"] for x in outputs]).mean()
        # test_f1_mean = torch.stack([x["test_f1"] for x in outputs]).mean()

        tqdm_dict = {"test_loss": test_loss_mean, "test_acc": test_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            # "test_loss": test_loss_mean,
            # "test_f1":test_f1_mean
        }
        return result

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [{"params": self.classification_head.parameters()}]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []


    def __retrieve_dataset(self, train=True, val=True, test=True):
        """Retrieves task specific dataset"""
        if train:
            return self.dataset.load_dataset(hparams.train_csv)
        elif val:
            return self.dataset.load_dataset(hparams.dev_csv)
        elif test:
            return self.dataset.load_dataset(hparams.test_csv)
        else:
            print("Incorrect dataset split")

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            # drop_last= True,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            # drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            # drop_last=True,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:

        parser.opt_list(
             "--learning_rate",
            default=7e-05,
            type=float,
            help="Classification head learning rate.",
        )

        # Data Args:

        parser.add_argument(
            "--train_csv",
            default="/data/newdisk/mrj_data/resistance/data/train_data_1.pkl",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="/data/newdisk/mrj_data/resistance/data/val_data_1.pkl",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="/data/newdisk/mrj_data/resistance/data/test_data_1.pkl",
            type=str,
            help="Path to the file containing the dev data.",
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


# 4. Create the experiments folder

def setup_testube_logger() -> TestTubeLogger:
    """Function that sets the TestTubeLogger to be used."""
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

    logger = TestTubeLogger(
        save_dir="experiments/",
        version=dt_string,
        name="lightning_logs",
        debug=True,
        log_graph=True,
    )

    return logger


logger = setup_testube_logger()

# 5.Create traininng arguments

# these are project-wide arguments
parser = HyperOptArgumentParser(
    strategy="random_search",
    description="Minimalist ProtAlbert Classifier",
    add_help=True,
)
parser.add_argument("--seed", type=int, default=3, help="Training seed.")
parser.add_argument(
    "--save_top_k",
    default=1,
    type=int,
    help="The best k models according to the quantity monitored will be saved.",
)
# Early Stopping
parser.add_argument(
    "--monitor", default="val_acc", type=str, help="Quantity to monitor."
)
parser.add_argument(
    "--metric_mode",
    default="max",
    type=str,
    help="If we want to min/max the monitored quantity.",
    choices=["auto", "min", "max"],
)
parser.add_argument(
    "--patience",
    default=60,
    type=int,
    help=(
        "Number of epochs with no improvement " "after which training will be stopped."
    ),
)
parser.add_argument(
    "--min_epochs",
    default=1,
    type=int,
    help="Limits training to a minimum number of epochs",
)
parser.add_argument(
    "--max_epochs",
    default=500,
    type=int,
    help="Limits training to a max number number of epochs",
)

# Batching
parser.add_argument("--batch_size", default=128, type=int, help="Batch size to be used.")
parser.add_argument(
    "--accumulate_grad_batches",
    default=32,
    type=int,
    help=(
        "Accumulated gradients runs K small batches of size N before "
        "doing a backwards pass."
    ),
)
parser.add_argument("--gpus", type=int, default=1, help="How many gpus")

parser.add_argument(
    "--val_percent_check",
    default=1.0,
    type=float,
    help=(
        "If you don't want to use the entire dev set (for debugging or "
        "if it's huge), set how much of the dev set you want to use with this flag."
    ),
)

# mixed precision
parser.add_argument(
    "--precision", type=int, default="32", help="full precision or mixed precision mode"
)
parser.add_argument("--amp_level", type=str, default="O3", help="mixed precision type")

# each LightningModule defines arguments relevant to it
parser = ProtAlbertClassifier.add_model_specific_args(parser)

hparams = parser.parse_known_args()[0]


# 6.Main Training steps

seed_everything(hparams.seed)

model = ProtAlbertClassifier(hparams)

early_stop_callback = EarlyStopping(
    monitor=hparams.monitor,
    min_delta=0.0,
    patience=hparams.patience,
    verbose=True,
    mode=hparams.metric_mode,
)

ckpt_path = os.path.join(
    logger.save_dir,
    logger.name,
    f"version_{logger.version}",
    "checkpoints",
)
# initialize Model Checkpoint Saver
checkpoint_callback = ModelCheckpoint(
    filepath=ckpt_path + "/" + "{epoch}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=hparams.save_top_k,
    verbose=True,
    monitor=hparams.monitor,
    period=1,
    mode=hparams.metric_mode,
)

trainer = Trainer(
    gpus=hparams.gpus,
    logger=logger,
    early_stop_callback=early_stop_callback,
    distributed_backend="dp",
    max_epochs=hparams.max_epochs,
    min_epochs=hparams.min_epochs,
    accumulate_grad_batches=hparams.accumulate_grad_batches,
    val_percent_check=hparams.val_percent_check,
    checkpoint_callback=checkpoint_callback,
    precision=hparams.precision,
    amp_level=hparams.amp_level,
    deterministic=True,
)

# tensorboard --logdir "experiments/lightning_logs/"
# writer = SummaryWriter(log_dir="tb_logs")

trainer.fit(model)
trainer.test()
