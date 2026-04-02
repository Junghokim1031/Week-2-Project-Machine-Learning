# AI 기반 인지 능력 예측 솔루션 (Cognitive Performance Predictor)

이 프로젝트는 개인의 수면 습관, 업무 시간, 스트레스 등의 라이프스타일 데이터를 기반으로 **인지 능력 점수(Cognitive Performance Score)**를 예측하는 머신러닝 대시보드입니다. Streamlit을 활용하여 대화형 웹 인터페이스를 제공하며, 사전 학습된 모델(Random Forest 등)을 기반으로 점수를 예측합니다.

---

## 주요 기능
- **인지 능력 예측**: 수면 시간, 수면의 질, 스트레스 수준, 운동량/활동 지수, 정신 건강 상태 등의 데이터를 입력하면 모델이 즉각적으로 예상 인지 점수(100점 만점)를 도출합니다.
- **맞춤형 피드백 제공**: 예측된 점수 구간에 따라 상태 양호 / 보통 / 주의 필요 등의 경고 메시지 및 상태 조언을 출력합니다.
- **모델 의사결정 시각화**: 모델 예측에 가장 큰 영향을 미치는 입력 변수들의 Feature Importance(피처 중요도) 그래프를 대시보드를 통해 직관적으로 분석할 수 있습니다.
- **검증된 성능 지표**: 하이퍼파라미터 튜닝을 거친 앙상블 모델(XGBoost)을 통해 **R² Score 0.80, MAE 7.97, RMSE 9.94** 수준의 안정적이고 신뢰도 높은 예측 분석을 수행합니다.

---

## 기술 스택 (Tech Stack)
- **언어**: Python 3
- **프론트엔드 및 데이터 대시보드**: [Streamlit](https://streamlit.io/)
- **머신러닝 파이프라인**: Scikit-Learn, XGBoost (`joblib`을 이용한 모델 및 전처리 객체 로드)
- **데이터 처리**: Pandas, NumPy
- **시각화 (학습 시)**: Matplotlib, Seaborn

---

## 📂 프로젝트 구조 (Directory Structure)
```bash
Week 2/
├── app.py                      # Streamlit 웹 대시보드 메인 실행 파일
├── dataset/                    # 원본 데이터셋 폴더
│   └── Sleep_health_and_lifestyle_dataset.csv
├── models/                     # 사전 학습된 머신러닝 모델 및 스케일러
│   ├── sleep_model_two.pkl
│   └── sleep_scaler_two.pkl
├── img/                        # 대시보드에 사용되는 시각화 차트 이미지
│   └── FeatureImportance.png
├── SleepHealthandLifestyle.ipynb # 데이터 탐색(EDA) 및 모델 학습 관련 노트북
├── sleepHealth.ipynb           # 수면 건강 및 라이프스타일 분석 관련 노트북
└── README.md                   # 프로젝트 설명 문서 (현재 파일)
```

---

## 설치 및 실행 방법 (How to Run)

1. **필수 라이브러리 설치**
   앱을 실행하기 위해 필요한 Python 패키지들을 설치합니다. 환경에 따라 로컬 또는 가상환경(예: `.venv`)에서 실행하세요.
   ```bash
   pip install streamlit pandas scikit-learn matplotlib seaborn joblib xgboost
   ```

2. **애플리케이션(Streamlit) 실행**
   터미널(Terminal) 앱에서 프로젝트 루트 디렉토리(`Week 2` 폴더)로 이동한 뒤 아래 명령어를 입력합니다.
   ```bash
   streamlit run app.py
   ```

3. **웹 브라우저에서 대시보드 확인**  
   실행 후 터미널에 출력되는 `Local URL` (기본값: `http://localhost:8501`)로 자동 접속되거나, 접속하여 대시보드 화면을 이용할 수 있습니다.

---

## 입력 변수 (Features)

**수치형 데이터 (Numerical Inputs)**
- 수면 시간 (0 ~ 24시간)
- 수면의 질 (0.0 ~ 10.0 점)
- 스트레스 수준 (0.0 ~ 10.0 점)
- 그날의 하루 근무 시간 (0 ~ 24시간)

**범주형 데이터 (Categorical Inputs)**
- **충분한 휴식 느낌 (Felt Rested)**: 네 / 아니오
- **요일 구분 (Day Type)**: 주중 / 주말
- **정신 건강 상태 (Mental Health)**: 건강 / 불안함 / 우울증 / 둘 다 복합적
- **수면 장애 위험 (Sleep Disorder Risk)**: 없음 / 낮음 / 중간 / 높음

*(입력된 범주형 데이터는 자동으로 One-Hot Encoding 등으로 처리되어 스케일러와 결합되어 계산됩니다.)*

---
*본 프로젝트는 수면 관련 요인 및 일상생활 데이터가 다음날 사람의 인지/작업 능력에 미치는 영향을 분석해주는 AI 기반 프로토타입 앱 프로젝트입니다.*
