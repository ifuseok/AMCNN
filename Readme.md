## Attention Based multi-channel CNN code source
[딥러닝 기술을 활용한 차별 및 혐오 표현 탐지 : 어텐션 기반 다중 채널 CNN 모델링](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002664602) 

위 논문에 사용한 모델 아키텍쳐 소스 코드 정fl



#### Test in NSMC Binary Classification
[네이버 영화평 Corpus](https://github.com/e9t/nsmc) 데이터 셋 활용 테스트 학습

    git clone 
    cd 
    git clone 
    python main.py ~~~~
    
#### Pre-Trained Embedding weight
[인터넷 뉴스 댓글 데이터 셋](https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments) 을 활용해 Word2Vec 을 학습하여 Pre-trained embedding으로 활용

#### requirements
* transformers == 3.x.x
* tensorflow >= 2.0.0
* keras >= 2.2.4
* emoji
* scikit-learn
* pandas
* gensim





#### References
* Pre-trained Weights Data :  https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments
* Tokenizers Reference : https://github.com/Beomi/KcBERT
* Model Architecture Base 논문 : [Multichannel CNN with Attention for Text Classification](https://arxiv.org/pdf/2006.16174.pdf)
