# Pstage_KLUE
## 🏆*최종 점수 및 등수*
- 점수 : 81.2
- 등수 : 7
## 🎈*문제 및 Data*
- 문장 내부의 두 객체 간의 관계를 대회에서 주어진 42개의 클래스로 분류하는 문제
- train(9000), test(1000)개의 data로 구성되어 있고, train data는 sentence, entity1, entity2, label이 포함되어 있다.
- 대회의 평가 지표는 Accuracy이다
## 📝*검증 전략*
- train data의 label분포를 확인해보니 클래스 불균형이 매우 심함 
  -> 9000개중 4432개가 관계_없음, 5개 미만으로 존재하는 label이 3가지..
- stratifiedkfold를 사용해 cross validation을 하려했는데 5개 미만으로 존재하는 label때문에 사용이 안됨 -> KFold 사용
- (문제점) KFold를 사용하니 특정 fold에서는 학습이 제대로 되지 않는 문제가 발생(valid accuracy가 증가하지 않음)
  - 데이터를 확인해보니 kfold.split과정에서 valid set에만 포함된 label이 다수 존재
  - 이를 해결 하기 위해 피어세션 팀원들과 직접 만든 추가 data를 사용해 stratifiedkfold사용
## *Data 추가*
- 피어세션 팀원들과 label의 개수가 가장 적은 하위 6개의 data를 각자 하나의 label을 담당해 직접 데이터를 추가 하였다.
- 많은 data를 추가하려는 것이 목표가 아닌 너무 적은 몇몇 label을 적당히 augmentation하기 위함이었기 때문에 wiki의 분류 기능을 사용해 쉽게 data를 추가하였다.
## *학습 방법*
- MODEL
  - hugging face에서 제공하는 pretrained model을 사용
    - monologg/koelectra-base-v3-discriminator
    - xlm-roberta-large
- Loss
  - CrossEntropy사용
- optimizer
  - SGD, Adam, AdamW등을 사용해보았는데 초반 koelectra model로 학습해본 결과 AdamW가 가장 좋아 AdamW고정해서 사용
- Ensemble
  - KFold를 통해 나온 모델들은 soft voting 방법을 사용하였고 최종 제출은 Koelectra model submission파일 3개와 xlm model submission파일 3개를 hard voting하여 사용하였다.
  - Ensemble을 통해 0.4점 점수를 올릴 수 있었다.
## 🖋️*시도해본 방법들*
### input format 변경하기
  최초 Baseline에서는 entity1[SEP]entity2[SEP]sentence의 format을 model input으로 사용하였는데 [SEP]토큰은 BERT pretrain에서 두개의 문장 사이 구분을 위해 사용하는데 위와 같은 방법보다는 새로운 token을 추가하던지, 아예 entity를 공백으로 만 구분할것이 좋다고 생각하였다. 앞의 생각과 오피스아워 시간 **이정우** 멘토님께서 제공해 주신 재미있는 Idea인 Sentence[SEP]"앞의 문장에서 entity1과 entity2는 어떤 관계야?" 느낌의 QA문제로 바라보는 방법도 고려하여 총 5가지 input format을 사용하였다.
  
    1. entity1[SEP]entity2[SEP]sentence
    2. entity1 entity2[SEP]sentence
    3. sentence[SEP]entity1[SEP]entity2
    4. sentence[SEP]앞의 문장에서 entity1랑 entity2는 문슨 관계야?
    5. [entity], [/entity] special token을 추가하여 단일 문장 classification Task수행
  - 1,2,3 번의 성능차이는 크지 않았음
  - 4번의 경우 tokenizer의 max_len을 100으로 설정하였을 때 1,2,3번에 비해 성능이 좋지 않았는데(5 ~ 8%차이) 이유는 질문 문장이 잘려 model이 entity두개를 제대로 받지 못한것으로 판단된다.
  - max_len을 256으로 설정하면 1,2,3,4의 성능은 모두 비슷하였다. 
  - 5번은 개인적으로 가장 성능이 잘 나올것이라 생각했던 방법인데 구현의 문제인지 hyper parameter의 문제인지 학습이 제대로 진행되지 않아 사용하지 못했다. 
### pororo를 사용한 Data augmentation
  - data를 다른 나라 언어로 번역한 뒤 다시 한국어로 번역하는 방식으로 Data를 추가하였다.
  - 학습 중 valid accuracy가 매우 높게나와 기대했지만 리더보드 점수는 기존의 점수에서 오히려 낮아졌다.
  - stage1에서도 겪었던 데이터 유출이 문제였는데 train set에 원본 데이터가 있고 valid set에 번역으로 추가한 data가 들어있어서 발생한 문제 같다.
  - 새로 추가된 데이터와 원본 데이터를 같은 set에 포함되도록 수정하여 학습해 보아도 결과 차이는 없을것이라 판단해 더 이상 사용하지 않았다.
### weighted CrossEntropy 사용
  - Imbalance문제를 해결하기 위해 많이 사용하는 Weighted CrossEntropy를 사용하였는데 큰 효과는 보지 못하였다.
  - test data자체도 매우 불균형 하기 때문이라고 예상이된다. 피어세션에서 test data의 50% 가량이 관계_없음 label이라는 소리를 들었는데 Weighted Loss를 사용하여 적은 label에 초점을 맞추기 보다는 test에 많이 존재하는 label에 초점을 맞추는게 리더보드 상으로는 높은 점수를 얻을 수 있을 것 같다.
### model 변경
  - 한국어 Task를 수행할 때 koelectra가 성능이 좋다고 하여 대회 종료 2일 전까지 하나의 모델만 사용하였는데 xlm모델이 성능이 좋다고 하여 model을 한번 변경해 보았다.
  - koelectra로 최고 점수는 77점 이었는데 xlm모델을 사용하자 바로 80점을 달성할 수 있었다. 
  - xlm모델로 변경하면서 memory문제로 max-len을 기존 256에서 190으로 줄이고, learning rate는 기존 5e-5에서 5e-6으로 줄여서 사용하였다.
## 😞아쉬운 점
- 지금 가장 아쉬운점은 input format 중 5번 방법을 빠르게 포기한것이다. hyper parameter를 고정해놓고 단 한번만 학습 돌려보고 바로 포기하였는데 조금 더 다양하게 시도해 보지 않은것이 후회된다.
- 너무 하나의 모델만 고집해서 사용한것이 아쉽다.
- 리더보드 점수만 고려해 학습한것 같아 아쉽다.
## 📈이번 대회를 통해 배운점
다양한 아이디어와 기술은 부수적인 것이고 해결하려는 문제에 가장 잘 맞는 모델을 찾는것이 가장 중요하다는것을 알게되었다. koelectra로 여러 방법을 사용해도 80의 벽을 넘기기 힘들었는데 xlm으로 변경하자마자 80이 넘는것을 보며 많은것을 느꼈다. 또한 대회를 통해 hugging face 문서를 읽어보며 hugging face를 사용할 수 있게 되었고, 이론으로만 배워 어떻게 해야할지 몰랐던 NLP Model의 fine tuning도 실습해보며 많은것을 배울 수 있었다.
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
- model의 input으로 어떤 format을 사용할지 정하는 인자
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
- input, output 파일 경로들은 아래와 같이 고정되어 있다.
- 실제 학습에 사용될 data는 train data path에 있는 file
- original data와 train data를 나눈 이유는 최초 제공 된 원본 파일은 변경하지 않고 augmentation, label수정을 적용해 새로운 data를 생성하기 위함
- original data path : /opt/ml/input/data/train/train2.tsv -> 최초 제공된 train file
- label type path : /opt/ml/input/data/label_type.pkl -> label file
- train data path : /opt/ml/input/data/train/newtrain.tsv -> 실제 학습에 사용될 file
- model save dir : 현재 dir에서 save_model dir안에 모델들이 저장
- test path : /opt/ml/input/data/test/test.tsv -> inference에 사용될 test file
- submission save dir : 현재 dir에서 prediction dir안에 submission.csv 파일이 생성
