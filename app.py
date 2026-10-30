import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from churn_pipeline import make_features

st.set_page_config(page_title='통신 고객 이탈 여부 예측 시스템', layout='wide')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_ml_objects():
    return joblib.load(ROOT / 'outputs_80_20/selected_model.joblib')

try:
    bundle = load_ml_objects()
    pipeline = bundle['pipeline']
    model = pipeline.named_steps['model']
except FileNotFoundError:
    st.error('학습된 모델 파일(joblib)을 찾을 수 없습니다. 먼저 jupyter notebook에서 모델을 학습하고 저장해주세요.')
    st.stop()

@st.cache_data
def load_raw_data():
    df = pd.read_csv(ROOT / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

df = load_raw_data()
st.title('📡 통신 고객 이탈 여부 예측 시스템')
st.write('고객의 세부 정보를 입력하여 이탈 가능성을 예측하고 통신 서비스 이용 현황을 분석합니다.')
main_tab1, main_tab2 = st.tabs(['이탈 위험도 시뮬레이터', '데이터 인사이트 대시보드'])

with main_tab1:
    st.markdown('### 개별 고객 이탈 예측')
    st.write("아래에 고객의 정보를 입력한 후 '예측 실행' 버튼을 눌러주세요.")
    col1, col2 = st.columns([2, 1.1])
    with col1:
        st.subheader('고객 정보 입력')
        with st.container(border=True):
            row = {}
            row['tenure'] = st.slider('가입 기간 (개월)', 0, 72, 12)
            a, b = st.columns(2)
            row['MonthlyCharges'] = a.number_input('월 청구 금액 ($)', 0.0, 500.0, 55.0)
            row['TotalCharges'] = b.number_input('누적 청구 금액 ($)', 0.0, 50000.0, 660.0)
            row['gender'] = a.selectbox('성별', ['Female', 'Male'], format_func=lambda x: '여성' if x == 'Female' else '남성')
            row['SeniorCitizen'] = b.selectbox('65세 이상 여부', ['0', '1'], format_func=lambda x: '예' if x == '1' else '아니오')
            row['Partner'] = a.selectbox('배우자 여부', ['Yes', 'No'], format_func=lambda x: '있음' if x == 'Yes' else '없음')
            row['Dependents'] = b.selectbox('부양가족 여부', ['No', 'Yes'], format_func=lambda x: '있음' if x == 'Yes' else '없음')
            row['Contract'] = a.selectbox('계약 유형', ['Month-to-month', 'One year', 'Two year'], format_func=lambda x: {'Month-to-month':'월 단위', 'One year':'1년 약정', 'Two year':'2년 약정'}[x])
            row['PaperlessBilling'] = b.selectbox('전자 청구서 사용 여부', ['Yes', 'No'], format_func=lambda x: '사용' if x == 'Yes' else '미사용')
            row['PaymentMethod'] = st.selectbox('결제 방식', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], format_func=lambda x: {'Electronic check':'전자 수표', 'Mailed check':'우편 수표', 'Bank transfer (automatic)':'은행 자동이체', 'Credit card (automatic)':'카드 자동결제'}[x])
            row['PhoneService'] = a.selectbox('전화 서비스 가입 여부', ['Yes', 'No'], format_func=lambda x: '가입' if x == 'Yes' else '미가입')
            row['MultipleLines'] = b.selectbox('복수 회선 사용 여부', ['No', 'Yes'] if row['PhoneService'] == 'Yes' else ['No phone service'], format_func=lambda x: {'Yes':'사용', 'No':'미사용', 'No phone service':'전화 미가입'}[x])
            row['InternetService'] = st.selectbox('인터넷 서비스', ['DSL', 'Fiber optic', 'No'], format_func=lambda x: {'DSL':'DSL', 'Fiber optic':'광인터넷', 'No':'미가입'}[x])
            for i, (key, label) in enumerate([('OnlineSecurity','온라인 보안'), ('OnlineBackup','온라인 백업'), ('DeviceProtection','기기 보호'), ('TechSupport','기술 지원'), ('StreamingTV','TV 스트리밍'), ('StreamingMovies','영화 스트리밍')]):
                row[key] = (a if i % 2 == 0 else b).selectbox(label, ['No','Yes'] if row['InternetService'] != 'No' else ['No internet service'], format_func=lambda x: {'Yes':'가입', 'No':'미가입', 'No internet service':'인터넷 미가입'}[x])
    with col2:
        st.subheader('예측 결과')
        st.write('')
        input_df = pd.DataFrame([row], columns=bundle['input_columns'])
        # 저장된 파이프라인에 인코딩과 스케일링이 포함되어 있습니다.
        input_features = make_features(input_df, bundle['variant'])
        with st.container(border=True):
            if st.button('이탈 여부 예측 실행', use_container_width=True, type='primary'):
                probability = float(pipeline.predict_proba(input_features)[0, 1])
                prediction = int(probability >= bundle['threshold'])
                st.metric(label='예측 상태 결과', value='이탈 위험' if prediction == 1 else '이용 유지')
                st.markdown('---')
                if prediction == 1:
                    st.error(f'**이 고객은 이탈 위험으로 예측되었습니다.**\n\n(이탈 확률: **{probability * 100:.2f}%**)')
                else:
                    st.success(f'**이 고객은 이용 유지로 예측되었습니다.**\n\n(유지 확률: **{(1 - probability) * 100:.2f}%**)')
            else:
                st.info('왼쪽 정보를 채우신 후 버튼을 누르면 AI 분석 결과가 표시됩니다.')

with main_tab2:
    st.markdown('### 통신 고객 데이터 시각화 및 인사이트')
    st.write('전체 데이터셋 분석을 기반으로 모델의 주요 변수와 고객 이용 현황 분포를 확인합니다.')
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(['AI 특성 중요도 분석', '요금 및 계약 유형 분석', '가입 기간 분포'])
    with sub_tab1:
        st.subheader('이탈 예측에 사용되는 주요 피처 (Feature Importance)')
        st.caption('모델의 이탈 여부 판단에 사용되는 상위 10개 피처 변수입니다.')
        names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importance = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x=importance.values, y=importance.index, hue=importance.index, legend=False, ax=ax, palette='viridis')
        ax.set_title('Top 10 Feature Importances', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
    with sub_tab2:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader('이탈 여부에 따른 월 청구 금액')
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.boxplot(x='Churn', y='MonthlyCharges', data=df, order=['No','Yes'], hue='Churn', palette='Set2', legend=False, ax=ax)
            ax.set_xticks([0,1], ['유지 (0)','이탈 (1)'])
            ax.set_title('Monthly Charges vs Churn')
            st.pyplot(fig)
            plt.close(fig)
        with col_t2:
            st.subheader('계약 유형에 따른 이탈 발생 현황')
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.countplot(x='Contract', hue='Churn', data=df, hue_order=['No','Yes'], palette='coolwarm', ax=ax)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ['유지','이탈'], title='이탈 여부')
            ax.set_title('Contract vs Churn')
            st.pyplot(fig)
            plt.close(fig)
    with sub_tab3:
        st.subheader('가입 기간별 이탈 고객 적층 분포')
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_df = df.assign(이탈여부=df.Churn.map({'No':'유지','Yes':'이탈'}))
        sns.histplot(data=plot_df, x='tenure', hue='이탈여부', hue_order=['유지','이탈'], multiple='stack', ax=ax, bins=10, palette='magma')
        ax.set_title('Tenure Distribution')
        st.pyplot(fig)
        plt.close(fig)
