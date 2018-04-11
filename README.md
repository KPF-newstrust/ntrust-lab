# NewsTrust LAB

NewsTrust deep learning algorithms


## Classification

* training file (category|형태소들)
```
1|유사시 함정 북한 지상시설 타격 전술 대지유도탄 탑재 신형 호위함 건조 해군 거
2|수도 아파트값 상승세 이어간 반면 지방 하락세 지속 한국감정원 주간 아파트 가격동
```

* train
```
python classify_rnn.py --train file_path_to_train --model_path dir_to_save_model --test file_path_to_test
```

* predict
```
python classify_rnn.py --predict doc_path_to_predict --model_path dir_to_load_model 
```


## Clustering

* training file (category|기사)
```
1|충북도가 구제역ㆍAI청정 지역을 끝까지 지켜냈다.
3|단양지역자활센터가 2014~2015년 2년 연속 '우수기관'으로 선정돼 운영비 1400만원을 지원 받는다.
```

* training word2vec
```
python train_w2v.py --data csv_path_to_train --model_path dir_to_save_model
```

* clustering
```
python dbscan_w2v.py --data csv_path_to_train --model_path dir_to_save_model --w2v_path w2v_dir_to_load
```


## NER

* training file (형태소\tTAG\tBIO)
```
할부	NNG	O
금융	NNG	O
서비스	NNG	O
업체	NNG	O
현대캐피탈	NNP	B-OG
과	JC	O
전업	NNG	O
카드	NNG	O
사	NR	O
현대	NNG	B-OG
카드	NNG	I-OG
```

* train
```
python ner.py --train file_path_to_train --model_path dir_to_save_model --w2v_path w2v_dir_to_load --test file_path_to_test
```

* predict
```
python ner.py --predict doc_path_to_predict --model_path dir_to_load_model --w2v_path w2v_dir_to_load
```
