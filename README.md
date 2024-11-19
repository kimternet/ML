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
## 3. 로지스틱 회귀(Logistic Regression)
  - 로지스틱 회귀 개념부터 다양한 데이터 적용, 확률적 경사 하강법(Stochastic Gradient Descent)
## 4. k최근접 이웃(k Nearest Neighbor)
  - k Nearest Neighbor을 이용한 분류와 회귀
## 5. 나이브 베이즈(Naive Bayes)
  - 가우시안(Gaussian), 베르누이(Bernoulli), 다항(Multinomial)
## 6. 서포트 벡터 머신(Support Vector Machine)
  - 서포트 벡터 개념과 커널 기법 SVR, SVC 사용, 다양한 데이터를 이용한 SVR, SVC 사용
## 7. 결정 트리(Decision Tree)
  - 의사결정 분석, 지니 불순도 Gini Index, 정보 획득량 Information Gain, 분산 감소
## 8. 앙상블(Ensemble)
  - Bagging 분류와 회귀, 랜덤 포레스트(Random Forest), 에이다 부스트(Ada Boost), 그레디언트 트리 부스팅(Gradient Tree Boosting), 스택(Stack), 보팅(Voting)
## 9. XGBoost, LightGBM
  - XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor
## 10. 군집화(Clustering)
  - K-Means, 미니 배치, 스펙트럼 군집화, 계층 군집화, DBSCAN, OPTICS, BIRCH
## 11. 다양체 학습(Manifold Learning)
  - 차원 축소, t-SNE, MDS, LLE, LTSA, Hessian, Isomap Modified LLE, SE, 원본 데이터와 정제된 데이터 비교
## 12. 분해(Decomposition)
  - PCA, SVD, NMF, 행렬 분해(Matrix Factorization)요인 분석, LDA
## 13. 추천 시스템(Recommender System)
  - 컨텐츠 기반 (Content- based Filtering), 협업 필터링(Collaborative Filtering), Hybrid 방식, SVD, SVD++, NMF
