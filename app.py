# c:/Workspace/Projects/Week 2/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

#==================
# 기초 페이지 설정
#==================
st.set_page_config(
    page_title='AI 기반 수면 및 인지 능력 예측 솔루션',
    page_icon='🧠',
    layout='wide'
)

#==================
# 모델 및 데이터 로드
#==================
@st.cache_resource
def load_assets():
    # 인지 능력 예측용 (XGBoost)
    model_cog = joblib.load('./models/sleep_model_v1.pkl')
    scaler_cog = joblib.load('./models/sleep_scaler_v1.pkl')
    encoder_cog = joblib.load('./models/sleep_encoder_v1.pkl')
    
    # 수면의 질 예측용 (Random Forest)
    model_sleep = joblib.load('./models/sleep_model.pkl')
    scaler_sleep = joblib.load('./models/sleep_scaler.pkl')
    
    return model_cog, scaler_cog, encoder_cog, model_sleep, scaler_sleep

model_cog, scaler_cog, encoder_cog, model_sleep, scaler_sleep = load_assets()

#==================
# 사이드바 입력
#==================
def user_input_features():
    st.sidebar.markdown("### 개인 건강 정보")
    # 성별, 업무 시간, 신체 활동량을 모두 더미 데이터(기본값)로 대체합니다.
    dummy_gender = 1 # 남성 기본값
    dummy_physical_activity = 60.0 
    dummy_work_hours = 8.0 
    
    age = st.sidebar.number_input('나이', 10, 100, 30)
    bmi_category = st.sidebar.selectbox('BMI 카테고리', ['정상 체중', '과체중', '비만'])
    
    st.sidebar.markdown("### 활동 및 신체 지표")
    daily_steps = st.sidebar.slider('일일 걸음 수', 0, 20000, 8000)
    heart_rate = st.sidebar.slider('심박수 (평상시)', 40, 120, 75)
    
    st.sidebar.markdown("### 혈압 및 수면 정보")
    bp_sys = st.sidebar.slider('수축기 혈압', 80, 180, 120)
    bp_dia = st.sidebar.slider('이완기 혈압', 50, 120, 80)
    sleep_duration_hrs = st.sidebar.slider('수면 시간', 0.0, 24.0, 8.0)
    stress_score = st.sidebar.slider('스트레스 지수', 0.0, 10.0, 5.0)
    
    felt_rested_label = st.sidebar.selectbox('충분히 휴식한 느낌', ['네', '아니요'])
    mental_sel = st.sidebar.selectbox('정신 건강 상태', ['없음(건강)', '불안증', '우울증', '불안&우울증'])
    sleep_disorder = st.sidebar.selectbox('수면 장애 여부', ['없음', '있음(불면증/무호흡증)'])
    # 매핑 로직 (사이드바 입력이 있는 것들만)
    bmi_map = {'정상 체중': 1, '과체중': 2, '비만': 3}
    disorder_map = {'없음': 0, '있음(불면증/무호흡증)': 1}
    mental_map = {'없음(건강)': 0, '불안증': 1, '우울증': 2, '불안&우울증': 3}
    rested_map = {'네': 1, '아니요': 0}
    # 연산 처리
    map_score = (bp_sys + (2 * bp_dia)) / 3
    activity_norm = (dummy_physical_activity - 59.17) / 20.83
    steps_norm = (daily_steps - 6816.85) / 1617.92
    overall_activity = (activity_norm + steps_norm) / 2
    # 수면 모델 입력 데이터프레임 (dummy_gender 적용)
    sleep_features = pd.DataFrame({
        'Gender': [dummy_gender],
        'Age': [age],
        'BMI Category': [bmi_map[bmi_category]],
        'Heart Rate': [heart_rate],
        'Sleep Disorder': [disorder_map[sleep_disorder]],
        'Mean_Arterial_Pressure': [map_score],
        'Overall_Activity_Score': [overall_activity]
    })
    
    # 인지 모델용 피처 구성 (dummy_work_hours 적용)
    cog_base_data = {
        'sleep_duration_hrs': sleep_duration_hrs,
        'stress_score': stress_score,
        'work_hours_that_day': dummy_work_hours,
        'mental_health_condition': mental_map[mental_sel],
        'felt_rested': rested_map[felt_rested_label],
        'sleep_disorder_risk': disorder_map[sleep_disorder] * 2
    }
    return sleep_features, cog_base_data

sleep_features, cog_base_data = user_input_features()

#==================
# 메인 화면 구성
#==================
st.title('AI 기반 수면 및 인지 능력 대시보드')

tab1, tab2 = st.tabs(["AI 통합 예측", "데이터 지표"])

with tab1:
    col1, col2 = st.columns(2)
    
    # "st.rerun()"을 제거하고 버튼 클릭 시 상태가 변하도록 간소화합니다.
    if st.button('종합 AI 분석 시작', use_container_width=True):
        with st.spinner('분석 중...'):
            # 1. 수면 모델 예측 (Sleep Quality)
            scaled_sleep = scaler_sleep.transform(sleep_features)
            predicted_quality = model_sleep.predict(scaled_sleep)[0]
            
            # 2. 인지 모델 입력 구성
            cog_input_data = {
                'sleep_duration_hrs': [cog_base_data['sleep_duration_hrs']],
                'sleep_quality_score': [predicted_quality],
                'stress_score': [cog_base_data['stress_score']],
                'work_hours_that_day': [cog_base_data['work_hours_that_day']],
                'day_type': [1], 
                'mental_health_condition': [cog_base_data['mental_health_condition']],
                'sleep_disorder_risk': [cog_base_data['sleep_disorder_risk']],
                'felt_rested': [cog_base_data['felt_rested']]
            }
            cog_features = pd.DataFrame(cog_input_data)
            
            # 3. 인지 모델 전처리
            cat_cols = ['day_type', 'mental_health_condition', 'sleep_disorder_risk', 'felt_rested']
            num_cols = ['sleep_duration_hrs', 'sleep_quality_score', 'stress_score', 'work_hours_that_day']
            
            encoded_cats = encoder_cog.transform(cog_features[cat_cols])
            encoded_df = pd.DataFrame(encoded_cats, columns=encoder_cog.get_feature_names_out(cat_cols))
            X_final = pd.concat([cog_features[num_cols], encoded_df], axis=1)
            
            scaled_cog = scaler_cog.transform(X_final)
            prediction_cog = model_cog.predict(scaled_cog)[0]
            
            # 상태 저장
            st.session_state['sleep_quality'] = predicted_quality
            st.session_state['cog_score'] = int(prediction_cog)
            st.session_state['last_updated'] = pd.Timestamp.now().strftime('%H:%M:%S')

    # 결과 디스플레이
    if 'sleep_quality' in st.session_state:
        col1.metric("예측된 수면의 질", f"{st.session_state['sleep_quality']} / 10")
        col2.metric("예상 인지 점수", f"{st.session_state['cog_score']} / 100")
        
        st.markdown(f"**최근 업데이트 시간:** `{st.session_state['last_updated']}`")
        st.markdown("---")
        
        
    else:
        st.warning("분석 시작 버튼을 눌러주세요.")

with tab2:
    st.write("모델 입력용 데이터 프레임:")
    st.dataframe(sleep_features)
