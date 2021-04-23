# Pstage_KLUE

## *Code*
### my_train.py
- main()
  - argument parsing
  - training, inference 과정을 끝낸 뒤 리더보드 점수를 console 입력으로 요청한다.
  - 리더보드 점수 입력하면 현재 사용한 hyperparameter들을 log.csv로 저장한다.(append)
- train(args, device)
  - argument의 model로 huggingface tokenizer, model을 불러옴 
  - [entity][/entity] special token이 필요한 학습인 경우 embedding layer를 생성한다.
  - StratifiedKFold(5)를 사용해 모델을 학습
- get_pred(model, device, minibatch, option, model_type, entity_embedding_layer)
  - option, 모델 별로 다른 input을 model에 넣고 logit을 반환하는 함수 
### my_load_data.py
- fix_label(origin_data_path, output_data_path)
  - origin_data_path 파일의 잘못된 label을 수정해 output_data_path로 저장하는 함수
- class MyDataSet(Dataset)
  - DataFrame, tokenizer, SEP_TOKEN, max_len, option을 받아 data와 label을 반환하는 Dataset Class
### my_eval.py
- inference(model_name, model_type, option, device)
  - 5fold의 각  fold별 best valid acc model을 사용해 submission file 생성
### my_utils.py
- set_seed(seed)
  - random seed 설정
- set_device()
  - device 설정, 반환
### FocalLoss.py
- FocalLoss(nn.Module)
  - focal loss를 계산해 반환 
---
## *training*
```
python my_train.py
```
---
## *my_train argument*
### --model_name MODEL_NAME
- 사용할 모델, huggingface pretrained model을 사용하며 아래 3개 모델 사용 가능
- xlm : xlm-roberta-large
- bert : kykim/bert-kor-base
- koelectra : monologg/koelectra-base-v3-discriminator
- default : koelectra
### --epochs EPOCHS
- number of epochs
- default : 15
### --lr LR
- learning rate
- default : 5e-5
### --option OPTION
- model의 input으로 어떤 format으로 사용할지 정하는 인자
- 0 : entity1[SEP]entity2[SEP]sentence
- 1 : entity1 entity2[SEP]sentence
- 2 : sentence[SEP]entity1[SEP]entity2
- 3 : sentence[SEP]앞의 문장에서 entity1랑 entity2는 무슨 관계야?
- 4 : [entity], [/entity] special token으로 sentence내부의 entity들을 감싸줌 -> 단일 문장 classification
- default : 0
### --seed SEED
- random seed
- default : 42
### --max_len MAX_LEN
- max model input size
- default : 190 
### --fix_label FIX_LABEL
- train data의 잘못된 label을 수정할지 말지를 정하는 인자
- default : True 
### --batch_size BATCH_SIZE
- batch size
- default : 32
---
## *input,output path*
- input, output 파일 경로들은 아래와 같이 고정되어 있습니다.
- 실제 학습에 사용될 data는 train data path에 있는 file입니다.
- original data와 train data를 나눈 이유는 최초 제공 된 원본 파일은 변경하지 않고 augmentation, label수정을 적용해 새로운 data를 생성하기 위함입니다.
- original data path : /opt/ml/input/data/train/train2.tsv -> 최초 제공된 train file
- label type path : /opt/ml/input/data/label_type.pkl -> label file
- train data path : /opt/ml/input/data/train/newtrain.tsv -> 실제 학습에 사용될 file
- model save dir : 현재 dir에서 save_model dir안에 모델들이 저장됩니다.
- test path : /opt/ml/input/data/test/test.tsv -> inference에 사용될 test file
- submission save dir : 현재 dir에서 prediction dir안에 submission.csv 파일이 생성됩니다.
