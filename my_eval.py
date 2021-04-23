import pandas as pd
from my_train import get_pred
import torch.nn as nn
from my_load_data import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def inference(model_name, model_type, option, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 42
    models = []
    entity_embedding_layers = []
    sep = tokenizer.sep_token
    test_path = "/opt/ml/input/data/test/test.tsv"
    test_data = pd.read_csv(test_path, delimiter='\t', header=None)
    test_set = MyDataSet(test_data, tokenizer, option=option, sep_token=sep)
    test_loader = DataLoader(test_set, shuffle=False)
    if option == 4:
        add_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]", "[/ENTITY]"]})
    for i in range(5):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config).to(device)
        if option == 4:
            model.resize_token_embeddings(tokenizer.vocab_size + add_token_num)
            entity_embedding_layer = nn.Embedding(2, model.get_input_embeddings().embedding_dim).to(device)
            entity_embedding_layers.append(entity_embedding_layer)
        else:
            entity_embedding_layers.append(None)
        model.load_state_dict(torch.load(f"./save_model/{i}fold_bestacc.pt"))
        model.eval()
        models.append(model)
    output_pred = []
    with torch.no_grad():
        for data in test_loader:
            for i in range(len(models)):
                if i == 0:
                    pred = get_pred(models[i], device, data, option, model_type, entity_embedding_layers[i])
                else:
                    pred += get_pred(models[i], device, data, option, model_type, entity_embedding_layers[i])
            result = pred.argmax(dim=-1)
            output_pred.append(result.detach().cpu().item())
    output = pd.DataFrame(output_pred, columns=["pred"])
    output.to_csv("./prediction/submission.csv", index=False)
