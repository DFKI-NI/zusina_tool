import pickle
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import torch


def prepare_data(path):
    """
    Preprocesses the labeled data.
    :param path: path to data
    :return: all data, train data, test data
    """
    df = pd.read_csv(path + 'data.csv').reset_index(drop=True)
    df = df.drop_duplicates(subset=['windowed_3'])
    print('columns', df.columns)
    df = df[df['windowed_3'].apply(len) > 1]
    df = df[df['windowed_3'].apply(len) <= 1000]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.fillna(0)
    df.to_csv(path + 'shuffled.csv', index=False)
    split = int(df.shape[0] * 0.7)
    df_train = df[:split]
    df_test = df[split:]
    return df, df_train, df_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def tokenize(df, df_train, df_test, path, name_tokenizer="distilbert-base-uncased"):
    """
    Tokenizes the data and splits the data into training, validation and test data.
    :param df: all data
    :param df_train: the train data
    :param df_test: the test data
    :param path: path to the data
    :param name_tokenizer: model name
    :return: tokenizer, data_collator, train_dataset, val_dataset, test_dataset, id2label, label2id
    """
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = pd.Series(df.iloc[:, 1:].columns).to_dict()
    label2id = {v: k for k, v in id2label.items()}

    pickle.dump(id2label, open(path + 'id2label.pckl', 'wb'))
    pickle.dump(label2id, open(path + 'label2id.pckl', 'wb'))

    val_split = int(df_train.shape[0] * 0.8)
    train_embeddings = tokenizer(df_train[:val_split]["windowed_3"].to_list(), truncation=True, padding=True)
    val_embeddings = tokenizer(df_train[val_split:]["windowed_3"].to_list(), truncation=True, padding=True)
    test_embeddings = tokenizer(df_test["windowed_3"].to_list(), truncation=True, padding=True)

    train_labels = df_train[:val_split].iloc[:, 1:].to_numpy().astype(float).tolist()
    val_labels = df_train[val_split:].iloc[:, 1:].to_numpy().astype(float).tolist()
    test_labels = df_test.iloc[:, 1:].to_numpy().astype(float).tolist()

    train_dataset = Dataset(train_embeddings, train_labels)
    val_dataset = Dataset(val_embeddings, val_labels)
    test_dataset = Dataset(test_embeddings, test_labels)
    return tokenizer, data_collator, train_dataset, val_dataset, test_dataset, id2label, label2id
