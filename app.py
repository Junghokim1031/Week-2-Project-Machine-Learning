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
# 모델 및 데이터 로드 (Encoder 포함)
#==================
@st.cache_resource
def load_assets():
    # 저장한 최신 v1 파일들을 불러옵니다.
    model = joblib.load('./models/sleep_model_v1.pkl')
    scaler = joblib.load('./models/sleep_scaler_v1.pkl')
    encoder = joblib.load('./models/sleep_encoder_v1.pkl')
    return model, scaler, encoder

model, scaler, encoder = load_assets()

#==================
# 사이드바 입력
#==================
def user_input_features():
    # --- 1. 수치형 데이터 입력 (Numerical) ---
    sleepTime = st.sidebar.slider('수면 시간', 0.0, 24.0, 8.0)
    sleepQuality = st.sidebar.slider('수면의 질', 0.0, 10.0, 8.0)
    stress = st.sidebar.slider('스트레스', 0.0, 10.0, 5.0)
    hoursWorked = st.sidebar.slider('운동량/활동 지수', 0.0, 24.0, 8.0)
 
    # --- 2. 범주형 데이터 입력 (Categorical) ---
    feltRested_label = st.sidebar.selectbox('충분히 휴식한 느낌', ['네', '아니요'])
    day_type_sel = st.sidebar.selectbox('요일', ['주중', '주말'])
    mental_sel = st.sidebar.selectbox('정신 건강 상태', ['없음(건강)', '불안증', '우울증', '불안&우울증'])
    risk_sel = st.sidebar.selectbox('수면 장애 위험', ['없음', '낮음', '중간', '높음'])

    # 방금 보내주신 컬럼 목록을 확인해보니, 학습 데이터에 사용되었던 값들이
    # 문자열('Weekday')이 아니라 이미 숫자로 변환된 상태(1, 2, 0 등)였음을 알 수 있습니다!
    # OneHotEncoder가 해당 숫자들을 인식할 수 있도록 매핑을 숫자로 변경합니다.
    day_map = {'주중': 1, '주말': 2}
    mental_map = {'없음(건강)': 0, '불안증': 1, '우울증': 2, '불안&우울증': 3}
    risk_map = {'없음': 0, '낮음': 1, '중간': 2, '높음': 3} # 확인하신 0~3 인덱스에 맞게 매핑
    rested_map = {'네': 1, '아니요': 0}

    # --- 3. 데이터프레임 구성 (학습 때 정의한 컬럼명과 완벽히 동일하게!) ---
    data = {
        'sleep_duration_hrs': [sleepTime],
        'sleep_quality_score': [sleepQuality],
        'stress_score': [stress],
        'work_hours_that_day': [hoursWorked],
        'day_type': [day_map[day_type_sel]],
        'mental_health_condition': [mental_map[mental_sel]],
        'sleep_disorder_risk': [risk_map[risk_sel]],
        'felt_rested': [rested_map[feltRested_label]]
    }
    
    features = pd.DataFrame(data)
    return features
    

features = user_input_features()

#==================
# 메인 화면 구성
#==================
st.title('인지 능력 분석 대시보드')

# 탭 구성
tab1, tab2 = st.tabs(["인지 능력 예측", "모델 성능 지표"])

with tab1:    
    st.subheader("예측 결과 및 데이터 분석")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if 'pred' in st.session_state:
            st.metric(label="예상 인지 점수", value=f"{st.session_state['pred']} / 100")
            
            if st.session_state['pred'] >= 80:
                st.success("현재 인지 상태가 매우 양호합니다!")
            elif st.session_state['pred'] >= 50:
                st.warning("보통 수준입니다. 충분한 휴식이 도움이 될 수 있습니다.")
            else:
                st.error("주의가 필요한 상태입니다.")

    with col2:
        st.write("**모델 예측 기여도 분석**")
        st.image("./img/FeatureImportance.png", use_container_width=True)

    st.markdown("---") 
    st.info("""
        **분석 참고 사항:** XGBoost 모델이 학습한 피처 중요도 차트를 참고하여 어떤 생활 습관이 가장 결정적인지 확인하세요. 
    """)

    # 예측 실행 버튼
    if st.button('결과 예측하기'):
        with st.spinner('AI 분석 중...'):
            # 1. 수치형과 범주형 컬럼 분리
            num_cols = ['sleep_duration_hrs', 'sleep_quality_score', 'stress_score', 'work_hours_that_day']
            cat_cols = ['day_type', 'mental_health_condition', 'sleep_disorder_risk', 'felt_rested']
            
            # 2. 범주형 데이터 인코딩 (불러온 encoder 사용)
            encoded_cats = encoder.transform(features[cat_cols])
            encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
            
            # 3. 데이터 결합 (수치형 컬럼 뒤에 인코딩된 범주형 컬럼 붙이기 - 학습 순서와 동일!)
            X_final = pd.concat([features[num_cols], encoded_df], axis=1)
            
            # 4. 스케일링
            scaled_input = scaler.transform(X_final)
            
            # 5. 최종 예측 (XGBoost)
            prediction = model.predict(scaled_input)
            st.session_state['pred'] = int(prediction[0])
            st.rerun()
    

with tab2:
    st.subheader("모델 신뢰도 분석 (XGBoost 튜닝 결과)")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    
    m_col1.metric("R² Score (설명력)", "0.80") 
    m_col2.metric("MAE (평균 오차)", "7.97")
    m_col3.metric("RMSE", "9.94")         