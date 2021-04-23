import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def fix_label(origin_data_path, output_data_path):
    """
    잘못된 label 수정
    """
    bad_label = {"wikitree-55837-4-0-2-10-11": "단체:구성원",
                 "wikitree-62775-3-3-7-0-2": "단체:본사_도시",
                 "wikitree-12599-4-108-111-4-7": "관계_없음",
                 "wikipedia-25967-115-24-26-35-37": "관계_없음",
                 "wikipedia-16427-6-14-17-20-22": "관계_없음",
                 "wikipedia-16427-8-0-3-26-28": "관계_없음",
                 "wikitree-19765-5-30-33-6-8": "관계_없음",
                 "wikitree-58702-0-18-20-22-24": "관계_없음",
                 "wikitree-71638-8-21-23-15-17": "관계_없음",
                 "wikipedia-257-0-0-1-53-57": "관계_없음",
                 "wikipedia-23188-0-74-86-41-42": "단체:하위_단체",
                 "wikipedia-13649-28-66-70-14-24": "관계_없음",
                 "wikipedia-6017-8-20-26-4-7": "관계_없음"}
    data = pd.read_csv(origin_data_path, delimiter='\t', header=None)
    data_l = list(bad_label.keys())
    for a in data_l:
        data.loc[data[data[0] == a].index, 8] = bad_label[a]
    data.to_csv(output_data_path, sep='\t', index=False, header=None)


class MyDataSet(Dataset):
    def __init__(self, data, tokenizer, label_dict=None, option=0, sep_token="[SEP]", max_len=190):
        """
        option 0 -> entity1[sep]entity2[sep]sentence
               1 -> entity1 entity2[sep]sentence
               2 -> sentence[sep]entity1[sep]entity2
               3 -> sentence[sep]"앞의 문장에서 entity1랑 entity2는 무슨 관계야?"
               4 -> se[entity]nt[/entity]en[entity]c[/entity]e -> add special token, entity embedding layer
        """
        self.data = data.reset_index(drop=True)
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.option = option
        self.sep = sep_token
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def input_style(self, sentence, entity1, entity2):
        if self.option == 4:
            sentence = sentence.replace(entity1, "[ENTITY]" + entity1 + "[/ENTITY]")
            sentence = sentence.replace(entity2, "[ENTITY]" + entity2 + "[/ENTITY]")
            tokenized_sentence = self.tokenizer(sentence, return_tensors="pt", padding="max_length",
                                                truncation=True, max_length=self.max_len, add_special_tokens=True)
            entity_token = []
            temp = []
            flag = 0
            tok_sent = self.tokenizer.tokenize(sentence, padding="max_length", truncation=True, max_length=self.max_len,
                                               add_special_tokens=True)
            for tok in tok_sent:
                if tok == "[/ENTITY]":
                    flag = 0
                temp.append(flag)
                if tok == "[ENTITY]":
                    flag = 1
            entity_token.append(temp)
            entity_token = torch.tensor(entity_token)
            tokenized_sentence['entity_token'] = entity_token
            return {key: val.squeeze() for key, val in tokenized_sentence.items()}

        if self.option == 0:
            temp = entity1 + self.sep + entity2
        elif self.option == 1:
            temp = entity1 + " " + entity2
        elif self.option == 2:
            temp = sentence
            sentence = entity1 + self.sep + entity2
        elif self.option == 3:
            temp = sentence
            sentence = "앞의 문장에서 " + entity1 + "랑 " + entity2 + "는 무슨 관계야?"

        tokenized_sentence = self.tokenizer(temp, sentence, return_tensors="pt", padding="max_length", truncation=True,
                                            max_length=self.max_len, add_special_tokens=True)
        return {key: val.squeeze() for key, val in tokenized_sentence.items()}

    def __getitem__(self, idx):
        """
        1 : sentence
        2 : entity1
        3 : entity1 start index
        4 : entity1 end index
        5 : entity2
        6 : entity2 start index
        7 : entity2 end index
        8 : label_string
        """
        cur_data = self.data.loc[idx]
        input_data = self.input_style(cur_data[1], cur_data[2], cur_data[5])
        if self.label_dict:
            label = self.label_dict[cur_data[8]]
            return input_data, label
        return input_data
