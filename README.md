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
2. K 최근접 이웃 회
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
## 7. 결정 트리(Decision Tree)
  - 의사결정 분석, 지니 불순도 Gini Index, 정보 획득량 Information Gain, 분산 감소
1. 분류를 위한 데이터
2. 회귀를 위한 데이터
3. 분류 - DecisionTreeClassifier()
   - 교차 검증
    - 전처리 없이 학습
    - 전처리 후 학습
   - 학습된 결정 트리 시각화
     - 텍스트를 통한 시각화
     - plot_tree를 사용한 시각화
     - graphviz를 사용한 시각화
   - 시각화
     - 결정경계 시각화
     - 하이퍼파라미터를 변경해 보면서 결정 경계의 변화 확인
4. 회귀 -DecisionTreeRegressor()
    - 교차 검증
      - 전처리 없이 학습
      - 전처리 후 학습
    - 학습된 결정 트리 시각화
      - 텍스트를 통한 시각화
      - plot_tree를 사용한 시각화
      - graphviz를 사용한 시각화
    - 시각화
      - 회귀식 시각화
      - 하이퍼파라미터를 변경해보면서 회귀식 시각
## 8. 앙상블(Ensemble)
  - Bagging 분류와 회귀, 랜덤 포레스트(Random Forest), 에이다 부스트(Ada Boost), 그레디언트 트리 부스팅(Gradient Tree Boosting), 스택(Stack), 보팅(Voting)
1. Bagging meta-estimator
   - Bagging을 사용한 분류
     - KNN
     - SVC
     - Decision Tree
   - Bagging을 사용한 회귀
     - KNN
     - SVR
     - Decision Tree
  - Forests of randomized trees
    - Random Forests 분류
    - Random Forests 회귀
    - Extremely Randomized Trees 분류
    - Extremely Randomized Trees 회귀
    - Random Forest, Extra Tree 시각화
  - AdaBoost
    - AdaBoost 분류
    - AdaBoost 회귀
  - Gradient Tree Boosting
    - Gradient Tree Boosting 분류
    - Gradient Tree Boosting 회귀
  - 투표 기반 분류(Voting Classifier)
    - 결정 경계 시각화
  - 투표 기반 회귀(Voting Regressor)
    - 회귀식 시각화
  - 스택 일반화(Stacked Generalization)
    - 스택 회귀
      - 회귀식 시각화
    - 스택 분류
      - 결정 경계 시각
## 9. XGBoost, LightGBM
  - XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor
1. XGBoost
   - 파이썬 기반 XGBoost
   - XGBClassifier
   - XGBRegressor
2. LIGHTGBM
   -LGBMClassifier
   -LGBMRegressor
## 10. 군집화(Clustering)
  - K-Means, 미니 배치, 스펙트럼 군집화, 계층 군집화, DBSCAN, OPTICS, BIRCH
1. 데이터 생성
2. K-평균(K-Means)
3. 미니 배치 K-평균(Mini Batch K-Means)
4. Affinity Propagation
5. Mean Shift
6. 스펙트럼 군집화(Spectral Clustering)
7. 계층 군집화(Hierarchical Clustering)
8. DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
9. OPTICS(Ordering Points To Identify the Clustering Structure)
10. Birch(Balanced iterative reducing and clustering using hierarchies)
11. 손글씨 데이터 군집화
    - K-MEANS
    - Spectral Clustering
    - Hierarchical Clustering
    - Birch
## 11. 다양체 학습(Manifold Learning)
  - 차원 축소, t-SNE, MDS, LLE, LTSA, Hessian, Isomap Modified LLE, SE, 원본 데이터와 정제된 데이터 비교
1. 데이터 생성 및 시각화 함수
2. Locally Linear Embedding(LLE)
3. Local Tangent Space Alignment(LTSA)
4. Hessian Eigenmapping
5. Modified Locally Linear Embedding
6. Isomap
7. Multi-Dimensional Scaling(MDS)
8. Spectral Embedding
9. t-distribued Stochastic Neighbor Embedding(t-SNE)
10 정제된 표현을 이용한 학습
   - 원본 데이터를 사용할 때
     - KNN
     - SVM
     - Decision Tree
     - Random Forest
   - 정제된 데이터를 사용할 때
     - KNN
     - SVM
     - Decision Tree
     - Random Forest
## 12. 분해(Decomposition)
  - PCA, SVD, NMF, 행렬 분해(Matrix Factorization)요인 분석, LDA
1. 데이터 불러오기 및 시각화
2. Principal Component Analysis(PCA)
3. Incremental PCA
4. Kernel PCA
5. Sparse PCA
6. Truncated Singular Value Decomposition(Truncated SVD)
7. Dictionary Learning
8. Factor Analysis
9. Independent Component Analysis(ICA)
10. Non-negative Matrix Factorization
11. Latent Dirichlet Allocation(LDA)
12. Linear Discriminant Analysis(LDA)
13. 압축된 표현을 사용한 학습
    - KNN
    - SVM
    - Decision Tree
    - Random Forest
14. 복원된 표현을 사용한 학습
    - KNN
    - SVM
    - Decision Tree
    - Random Forest
15. 이미지 복
## 13. 추천 시스템(Recommender System)
  - 컨텐츠 기반 (Content- based Filtering), 협업 필터링(Collaborative Filtering), Hybrid 방식, SVD, SVD++, NMF
1. Surprise
2. 컨텐츠 기반 필터링(Content-based Filtering)
3. 협업 필터링(Collaborative Filtering)
4. 하이브리드 (Hybrid)
