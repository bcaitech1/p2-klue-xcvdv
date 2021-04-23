from my_load_data import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import StratifiedKFold
from my_utils import *
from my_eval import *
from FocalLoss import *
import pickle as pickle
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm

model_dict = {"koelectra": "monologg/koelectra-base-v3-discriminator",
              "bert": "kykim/bert-kor-base",
              "xlm": "xlm-roberta-large"}


def get_pred(model, device, minibatch, option, model_type, entity_embedding_layer):
    if option == 4:
        entity_token = entity_embedding_layer(minibatch["entity_token"].to(device))
        input_embed = (model.get_input_embeddings())(minibatch["input_ids"].to(device))
        input_embed += entity_token
        if model_type == "xlm":
            pred = model(inputs_embeds=input_embed.to(device),
                         attention_mask=minibatch["attention_mask"].to(device))[0]
        else:
            pred = model(inputs_embeds=input_embed.to(device),
                         token_type_ids=minibatch["token_type_ids"].to(device),
                         attention_mask=minibatch["attention_mask"].to(device))[0]
    elif model_type == "xlm":
        pred = model(input_ids=minibatch["input_ids"].to(device),
                     attention_mask=minibatch["attention_mask"].to(device))[0]
    else:
        pred = model(input_ids=minibatch["input_ids"].to(device),
                     token_type_ids=minibatch["token_type_ids"].to(device),
                     attention_mask=minibatch["attention_mask"].to(device))[0]
    return pred


def train(args, device):
    model_name = model_dict[args.model_name]
    epochs = args.epochs
    option = args.option
    learning_rate = args.lr
    batch_size = args.batch_size
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    data = pd.read_csv(args.train_data_path, delimiter='\t', header=None)
    labels = [label_type[x] for x in list(data.iloc[:, 8])]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if option == 4:
        add_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[ENTITY]", "[/ENTITY]"]})
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 42
    sep = tokenizer.sep_token

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    best_accs = []
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(data, labels)):
        train_set = MyDataSet(data.loc[train_idx, :], tokenizer, label_dict=label_type, option=option, sep_token=sep)
        valid_set = MyDataSet(data.loc[valid_idx, :], tokenizer, label_dict=label_type, option=option, sep_token=sep)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config).to(device)
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss()
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)

        if option == 4:
            entity_embedding_layer = nn.Embedding(2, model.get_input_embeddings().embedding_dim).to(device)
            model.resize_token_embeddings(tokenizer.vocab_size + add_token_num)
        entity_embedding_layer = None
        max_valid_acc = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            train_acc = 0
            train_loss = 0
            valid_acc = 0
            valid_loss = 0
            for minibatch, target in train_loader:
                target = target.to(device)
                pred = get_pred(model, device, minibatch, option, args.model_name, entity_embedding_layer)
                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pred_result = torch.argmax(pred.detach(), dim=1)
                train_loss += loss.item()
                train_acc += (target == pred_result).sum().item()

            # Validation
            model.eval()
            cur_label = []
            cur_pred = []
            with torch.no_grad():
                for minibatch, target in valid_loader:
                    target = target.to(device)
                    pred = get_pred(model, device, minibatch, option, args.model_name, entity_embedding_layer)
                    loss = criterion(pred, target)
                    pred_result = torch.argmax(pred.detach(), dim=1)
                    cur_label.extend(target.cpu().tolist())
                    cur_pred.extend(pred_result.cpu().tolist())
                    valid_loss += loss.item()
                    valid_acc += (target == pred_result).sum().item()

            f1score = metrics.f1_score(cur_label, cur_pred, average="macro")
            print(f"fold:{fold} train_loss:{train_loss/len(train_set):.4} train_acc:{train_acc/len(train_set):.4} "
                  f"valid_loss:{valid_loss/len(valid_set):.4} valid_acc:{valid_acc/len(valid_set):.4} f1:{f1score:.4}")
            # scheduler.step(valid_loss)
            if max_valid_acc < valid_acc/len(valid_set):
                torch.save(model.state_dict(), f"save_model/{fold}fold_bestacc.pt")
                max_valid_acc = valid_acc/len(valid_set)
        best_accs.append(max_valid_acc)
    print("best acc each fold : ", best_accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="koelectra", help="model(ex: xlm, bert default:koelectra)")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs(default:15)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate(default:5e-5)")
    parser.add_argument("--option", type=int, default=0, help="input option(read my_load_data.py)")
    parser.add_argument("--origin_data_path", type=str, default="/opt/ml/input/data/train/train2.tsv")
    parser.add_argument("--train_data_path", type=str, default="/opt/ml/input/data/train/newtrain.tsv")
    parser.add_argument("--seed", type=int, default=42, help="input random seed(default:42)")
    parser.add_argument("--max_len", type=int, default=190, help="input max model input size(default:190)")
    parser.add_argument("--fix_label", type=bool, default=True, help="fix data label(default:true)")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size(default:32)")
    args = parser.parse_args()

    device = set_device()
    set_seed(args.seed)
    if args.fix_label:
        fix_label(args.origin_data_path, args.train_data_path)
    print("train start")
    train(args, device)
    print("inference start")
    inference(model_dict[args.model_name], args.model_name, args.option, device)
    score = input("리더보드 점수 입력 : ")
    fields = ["MODEL_NAME", "option", "lr", "optimizer", "scheduler", "max_len", "fix_label", "batch_size",
              "epochs", "seed", "score"]
    if not os.path.exists("log.csv"):
        data = [(args.model_name, args.option, args.lr, "AdamW", "ReduceLR", args.max_len, args.batch_size,
                 args.epochs, args.seed, score)]
        df = pd.DataFrame(data, columns=fields)
    else:
        data = [args.model_name, args.option, args.lr, "AdamW", "ReduceLR", args.max_len, args.batch_size,
                args.epochs, args.seed, score]
        df = pd.read_csv("log.csv")
        df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    df.to_csv("log.csv", index=False)


if __name__ == "__main__":
    main()
