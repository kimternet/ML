## 1. 사이킷런(scikit-learn)
  - scikit-learn 주요 모듈, Estimator API, 데이터 전처리 모듈, 성능 평가 지표 학습
1. scikit-learn 특징
2. scikit-learn 주요 모듈
3. estimator API
   - API 사용 방법
   - API 사용 예제
4. 예제 데이터 셋
   - 분류 또는 회귀용 데이터셋
   - 온라인 데이터 셋
   - 분류와 클러스터링을 위한 표본 데이터 생성
   - 예제 데이터 셋 구조
5. model_selection 모듈
   - train_test_split(): 학습/테스트 데이터 셋 분리
   - cross_val_score(): 교차검증
   - GridSearchCV: 교차 검증과 최적 하이퍼 파라미터 찾기
6. preprocessing 데이터 전처리 모듈
   - StandardScaler: 표준화 클래스
   - MinMaxScaler: 정규화 클래스
7. 성능 평가 지표
   - 정확도(Accuracy)
   - 오차 행렬(Confusion Matrix)
   - 정밀도 (Precision)와 재현율(Recall)
   - F1 Score(F-measure)
   - ROC 곡선과 AUC
## 2. 선형 모델(Linear Models)
  - 선형 회귀 Linear Regression, 릿지 회귀 Ridge Regression, 라쏘 회귀 Lasso Regression, 신축망 ElasticNet, 직교 정합 추구 Orthogonal Matchin Pursuit, 다항 회귀 Polynomial Regression
1. 선형 회귀(Linear Regression)
   - 선형 회귀 예제
2. 릿지 회귀(Ridge Regression)
   - 릿지 회귀 예제
3. 라쏘 회귀(Lasso Regression)
   - 라쏘 회귀 예제
4. 신축망(Elastic-Net)
   - 신축망 예제
5. 직교 정합 추구(Orthogonal Matching Pursuit)
   - 직교 정합 예제
6. 다항 회귀(Polynomial Regression)
   - 다항 회귀 예제
## 3. 로지스틱 회귀(Logistic Regression)
  - 로지스틱 회귀 개념부터 다양한 데이터 적용, 확률적 경사 하강법(Stochastic Gradient Descent)
1. 로지스틱 회귀 예제
2. 확률적 경사 하강법(Stochastic Gradient Descent)
   - SGD를 사용한 선형 회귀 분석
   - 데이터에 대한 SGD 분류
## 4. k최근접 이웃(k Nearest Neighbor)
  - k Nearest Neighbor을 이용한 분류와 회귀
1. K 최근접 이웃 분류
2. K 최근접 이웃 회귀
## 5. 나이브 베이즈(Naive Bayes)
  - 가우시안(Gaussian), 베르누이(Bernoulli), 다항(Multinomial)
1. 나이브 베이즈 분류기의 확률 모델
2. 산림 토양 데이터
   - 학습,평가 데이터 분류
   - 전처리
     - 전처리 전 데이터
     - 전처리 과정
     - 전처리 후 데이터
3. 20 Newsgroup 데이터
   - 학습, 평가 데이터 분류
   - 벡터화
     - CountVectorizer
     - HashingVectorizer
     - TfidVectorizer
4. 가우시안 나이브 베이즈
5. 베르누이 나이브 베이즈
   - 학습 및 평가(Count,Hash,Tf-idf)
   - 시각화
6. 다항 나이브 베이즈
   - 학습 및 평가(Count, Tf-idf)
   - 시각화
## 6. 서포트 벡터 머신(Support Vector Machine)
  - 서포트 벡터 개념과 커널 기법 SVR, SVC 사용, 다양한 데이터를 이용한 SVR, SVC 사용
1. SVM을 이용한 회귀 모델과 분류 모델
   - SVM을 사용한 회귀 모델(SVR)
   - SVM을 사용한 분류 모델(SVC)
2. 커널 기법
3. 매개변수 튜닝
4. 데이터 전처리
5. Linear SVR
6. Kernel SVR
7. Linear SVC
8. Kernel SVC
## 7. 결정 트리(Deci원
## 13. 추천 시스템(Recommender System)
  - 컨텐츠 기반 (Content- based Filtering), 협업 필터링(Collaborative Filtering), Hybrid 방식, SVD, SVD++, NMF
1. Surprise
2. 컨텐츠 기반 필터링(Content-based Filtering)
3. 협업 필터링(Collaborative Filtering)
4. 하이브리드 (Hybrid)
