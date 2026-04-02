import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#==================
# 기초 페이지 설정
#==================
st.set_page_config(
    page_title='AI 기반 인지 능력 예측 솔루션',
    page_icon='🧠',
    layout='wide'
)

#==================
# 모델 및 데이터 로드
#==================
@st.cache_resource
def load_assets():
    model = joblib.load('./models/sleep_model_two.pkl')
    scaler = joblib.load('./models/sleep_scaler_two.pkl')
    # 히트맵이나 그래프를 위한 원본 데이터 샘플이 있다면 로드 (선택 사항)
    # df = pd.read_csv('sleep_data.csv') 
    return model, scaler

model, scaler = load_assets()

#==================
# 사이드바 입력 (기존 로직 유지)
#==================
def user_input_features():
    # --- 1. Basic Numerical Inputs ---
    sleepTime = st.sidebar.slider('수면 시간', 0.0, 24.0, 8.0)
    sleepQuality = st.sidebar.slider('수면의 질', 0.0, 10.0, 8.0)
    stress = st.sidebar.slider('스트레스', 0.0, 10.0, 5.0)
    hoursWorked = st.sidebar.slider('운동량/활동 지수', 0.0, 24.0, 8.0)
 
    # Felt Rested (Assuming it was binary 0/1)
    feltRested_label = st.sidebar.selectbox('충분히 휴식한 느낌', ['네', '아니요'])
    feltRested = 1 if feltRested_label == '네' else 0

    # --- 2. Categorical Selectboxes ---
    day_type_sel = st.sidebar.selectbox('요일', ['주중', '주말'])
    mental_sel = st.sidebar.selectbox('정신 건강 상태', ['건강', '불안함', '우울증', '둘다'])
    risk_sel = st.sidebar.selectbox('수면 장애 위험', ['없음', '낮음', '중간', '높음'])

    # --- 3. Construct the 15-column Dictionary ---
    data = {
        # Numerical
        'sleep_duration_hrs': sleepTime,
        'sleep_quality_score': sleepQuality,
        'stress_score': stress,
        'work_hours_that_day': hoursWorked,
        'felt_rested': feltRested,
        
        # Day Type One-Hot (주중=1, 주말=2)
        'day_type_1': 1 if day_type_sel == '주중' else 0,
        'day_type_2': 1 if day_type_sel == '주말' else 0,
        
        # Mental Health One-Hot (건강=0, 불안=1, 우울=2, 둘다=3)
        'mental_health_condition_0': 1 if mental_sel == '건강' else 0,
        'mental_health_condition_1': 1 if mental_sel == '불안함' else 0,
        'mental_health_condition_2': 1 if mental_sel == '우울증' else 0,
        'mental_health_condition_3': 1 if mental_sel == '둘다' else 0,
        
        # Sleep Disorder Risk One-Hot (없음=0, 낮음=1, 중간=2, 높음=3)
        'sleep_disorder_risk_0': 1 if risk_sel == '없음' else 0,
        'sleep_disorder_risk_1': 1 if risk_sel == '낮음' else 0,
        'sleep_disorder_risk_2': 1 if risk_sel == '중간' else 0,
        'sleep_disorder_risk_3': 1 if risk_sel == '높음' else 0,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features
    

features = user_input_features()

#==================
# 메인 화면 구성
#==================
st.title('인지 능력 분석 대시보드')

# 탭 구성: 예측 / 모델 성능 분석 / 데이터 통계
tab1, tab2 = st.tabs(["인지 능력 예측", "모델 성능 지표"])

with tab1:    
    st.subheader("예측 결과 및 데이터 분석")
    
    # 1. 화면을 좌(col1), 우(col2)로 분할
    col1, col2 = st.columns([1, 1])  # [1, 1.2] 처럼 숫자를 조절해 너비 비율 조정 가능

    with col1:
        # 좌측: 기존 예측 결과 메트릭 및 피드백 표시
        if 'pred' in st.session_state:
            st.metric(label="예상 인지 점수", value=f"{st.session_state['pred']} / 100")
            
            if st.session_state['pred'] >= 80:
                st.success("현재 인지 상태가 매우 양호합니다!")
            elif st.session_state['pred'] >= 50:
                st.warning("보통 수준입니다. 충분한 휴식이 도움이 될 수 있습니다.")
            else:
                st.error("주의가 필요한 상태입니다.")

    with col2:
        # 우측: 피처 중요도(Feature Importance) 그래프 삽입
        st.write("**모델 예측 기여도 분석**")
        # 이미지 파일 경로를 본인의 환경에 맞게 수정하세요 (예: 'FeatureImportance.png')
        st.image("./img/FeatureImportance.png", use_container_width=True)

    # 2. 하단에 분석 정보 알림(Disclaimer) 추가
    st.markdown("---") # 구분선
    st.info("""
        **분석 참고 사항:** 현재 모델에서 '수면의 질(Quality of Sleep)'이 가장 높은 비중을 차지하고 있으나, 
        이는 스트레스 지수, 수면 시간 등 다른 생활 습관 요인들이 복합적으로 반영된 결과입니다. 
        개별 변수 간의 상관관계는 위 지표에 직접적으로 드러나지 않을 수 있으므로 종합적인 해석이 필요합니다.
    """)

    # 예측 실행 버튼 (결과 창 아래에 배치)
    if st.button('결과 예측하기'):
        with st.spinner('AI 분석 중...'):
            scaled_input = scaler.transform(features)
            prediction = model.predict(scaled_input)
            st.session_state['pred'] = int(prediction[0])
            st.rerun() # 즉시 화면 갱신을 위해 추가 권장
    

with tab2:
    st.subheader("모델 신뢰도 분석")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    m_col1.metric("R² Score (설명력)", "0.93")
    m_col2.metric("MAE (평균 오차)", "4.61")
    m_col3.metric("RMSE", "33.29")