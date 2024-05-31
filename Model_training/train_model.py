# -*- coding: utf-8 -*-

import datetime
import os
import pickle

import numpy as np
import torch
from torchmetrics import F1Score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EvalPrediction

from Model_training import dataset

# Load and preprocess labeled text examples (training data)
path = ''
df, df_train, df_test = dataset.prepare_data(path)
tokenizer, data_collator, train_dataset, val_dataset, test_dataset, id2label, label2id = dataset.tokenize(df, df_train,
                                                                                                          df_test, path)
num_labels = len(df.iloc[:, 1:].columns)


def compute_metrics(p: EvalPrediction, thresh=0.3):
    """
    Computes the performance metrics.
    :param p: prediction
    :param thresh: threshold
    :return: f1score and accuracy
    """
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= thresh)] = 1
    metric = F1Score(num_labels=num_labels, task='multilabel')
    f1_score = metric(torch.tensor(y_pred), torch.tensor(p.label_ids))
    accuracy = np.mean(sum(y_pred == p.label_ids) / len(y_pred))
    return {'f1_score': f1_score, 'accuracy': accuracy}


# training
os.environ["TOKENIZERS_PARALLELISM"] = "true"
training_args = TrainingArguments(
    output_dir=path + 'results/transformer_model/',
    learning_rate=0.0001,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function.
        :param model: model
        :param inputs: input data
        :param return_outputs: whether to return the outputs
        :return: the loss
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = torch.sigmoid(logits)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss


# load pretrained model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                           problem_type="multi_label_classification",
                                                           num_labels=num_labels,
                                                           id2label=id2label,
                                                           label2id=label2id)
# train the model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Test the model
predictions = trainer.predict(test_dataset)
print('on test')
metrics = compute_metrics(predictions, thresh=0.3)
fscore = metrics['f1_score']
print(metrics)

# Save the model
pickle.dump(tokenizer, open(
    path + 'tokenizer' + datetime.datetime.today().strftime("_%m_%d_%H_%M") + '.pckl', 'wb'))
pickle.dump(model, open(
    path + 'model' + datetime.datetime.today().strftime("_%m_%d_%H_%M") + '.pckl', 'wb'))
