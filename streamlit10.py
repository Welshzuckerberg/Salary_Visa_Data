import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')

# 데이터 로드
salary_data_path = './Data/p_salary_data_total.csv'
salary_data_total_path = './Data/salary_data_total.csv'

salary_data = pd.read_csv(salary_data_path)
salary_data_total = pd.read_csv(salary_data_total_path)

# 경험 수준 한글 변환
experience_level_map = {
    'EN': '엔트리급',
    'MI': '중급',
    'SE': '시니어급',
    'EX': '고위급'
}

salary_data['experience_level'] = salary_data['experience_level'].map(experience_level_map)
salary_data_total['experience_level'] = salary_data_total['experience_level'].map(experience_level_map)

# 데이터프레임 생성
df = pd.concat([salary_data[['work_year', 'job_title', 'salary_in_usd', 'experience_level']], 
                salary_data_total[['work_year', 'job_title', 'salary_in_usd', 'experience_level']]])
df_grouped = df.groupby(['job_title', 'experience_level']).agg({'salary_in_usd': 'mean'}).reset_index()
df_grouped.columns = ['직업군', '경력 수준', '평균연봉']

# 평균연봉을 정수로 변환
df_grouped['평균연봉'] = df_grouped['평균연봉'].round().astype(int)

# 국가 데이터 추가
df_grouped['국가'] = 'USA'

# 비자 데이터 로드
visa_data_path = 'visa_data.csv'

# 앱의 제목 만들기
st.title(':rainbow[파린이 유랑단]:heart_eyes:')

# 세션 상태를 사용하여 상태 관리
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# 페이지 전환 함수
def go_to_page(page):
    st.session_state.page = page

# 메인 페이지
if st.session_state.page == 'main':
    st.markdown("""
                <div style="font-size:30px;"><직업별 연봉 예측 및 비자 발급 여부 예측></div>
                <br>
                <div style="font-size:20px;">직업을 선택하면 대략적인 연봉을 예측하는 프로그램입니다.</div>
                """, unsafe_allow_html=True)
    if st.button('직업 정보 제공'):
        go_to_page('page1')
       
    st.markdown('<div style="font-size:20px;">직업을 선택하면 대략적인 비자 발급 여부를 예측하는 프로그램입니다.</div>', unsafe_allow_html=True)
    if st.button('비자 정보 예측'):
        go_to_page('page2')

# 페이지 1
elif st.session_state.page == 'page1':
    st.title("AI/ML 연봉 데이터 제공")
    st.write("###### 원하는 직업군을 최대 5개까지 선택하세요.")
   
    job_selected = st.multiselect("직업 선택", df_grouped["직업군"].unique(), default=df_grouped["직업군"].unique()[:5])
    experience_selected = st.selectbox("경력 수준 선택", ['엔트리급', '중급', '시니어급', '고위급'])
   
    # 선택된 직업과 경력 수준에 대한 데이터 필터링
    job_data = df_grouped[(df_grouped["직업군"].isin(job_selected)) & (df_grouped["경력 수준"] == experience_selected)]

    # 데이터 출력
    st.write("### 선택한 직업에 대한 정보")
    st.write(job_data)

    # 시각화 1: 연봉 분포
    st.write("### 연봉 분포")
    fig, ax = plt.subplots()
    for job in job_selected:
        ax.hist(df[df["job_title"] == job]["salary_in_usd"], bins=20, alpha=0.5, label=job)
    ax.set_title("연봉 분포")
    ax.set_xlabel("연봉")
    ax.set_ylabel("빈도")
    ax.legend()
    st.pyplot(fig)

    # 시각화 2: 경력 수준별 평균 연봉
    st.write("### 경력 수준별 평균 연봉")
    fig, ax = plt.subplots()
    for job in job_selected:
        experience_data = df_grouped[(df_grouped["직업군"] == job)].sort_values(by="경력 수준")
        ax.bar(experience_data["경력 수준"], experience_data["평균연봉"], alpha=0.5, label=job)
    ax.set_title("경력 수준별 평균 연봉")
    ax.set_xlabel("경력 수준")
    ax.set_ylabel("평균 연봉")
    ax.legend()
    st.pyplot(fig)

    # 시각화 3: 연도별 평균 연봉 변화
    st.write("### 연도별 평균 연봉 변화")
    fig, ax = plt.subplots()
    for job in job_selected:
        yearly_data = df[df["job_title"] == job].groupby('work_year').agg({'salary_in_usd': 'mean'}).reset_index()
        ax.plot(yearly_data['work_year'], yearly_data['salary_in_usd'], marker='o', linestyle='-', label=job)
    ax.set_title("연도별 평균 연봉 변화")
    ax.set_xlabel("연도")
    ax.set_ylabel("평균 연봉")
    ax.legend()
    st.pyplot(fig)

    if st.button('뒤로'):
        go_to_page('main')

# 페이지 2
elif st.session_state.page == 'page2':
    # 비자 데이터 로드
    visa_data = pd.read_csv(visa_data_path)

    # 필요한 컬럼 선택 및 전처리
    visa_data = visa_data[['SOC_NAME', 'CASE_STATUS']].dropna()

    # CASE_STATUS를 이진 변수로 변환
    visa_data['CASE_STATUS'] = visa_data['CASE_STATUS'].apply(lambda x: 1 if x == 'CERTIFIED' else 0)

    # 특징과 타겟 변수 분리
    X = visa_data[['SOC_NAME']]
    y = visa_data['CASE_STATUS']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 원-핫 인코더 설정
    encoder = OneHotEncoder()

    # 로지스틱 회귀 모델 설정
    model = LogisticRegression()

    # 전처리와 모델을 포함한 파이프라인 설정
    pipeline = Pipeline([
        ('encoder', encoder),
        ('model', model)
    ])

    # 모델 학습
    pipeline.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(pipeline, 'visa_predictor.pkl')

    # SOC_NAME 리스트 추출
    soc_names = visa_data['SOC_NAME'].unique()

    # 저장된 모델 로드
    pipeline = joblib.load('visa_predictor.pkl')

    # 비자 발급 예측 함수
    def analyze_job(soc_name):
        # 입력 데이터를 변환 및 예측 수행
        prediction = pipeline.predict([[soc_name]])
        return 'CERTIFIED' if prediction[0] == 1 else 'DENIED'

    # Streamlit UI 구성
    st.title('비자 발급 여부 예측')
    st.write('직업을 입력하여 H1B의 발급 여부를 알 수 있는 프로그램입니다.')

    # 사용자로부터 키워드 입력 받기
    keyword = st.text_input('###### 직업 키워드를 입력해주세요:')

    # 키워드에 따른 SOC_NAME 필터링
    filtered_soc_names = [name for name in soc_names if keyword.lower() in name.lower()]

    # 필터링된 SOC_NAME을 드롭다운 메뉴로 제공
    soc_name_input = st.selectbox('###### 해당하는 직업을 선택해 주세요:', filtered_soc_names)

    # 사용자가 SOC_NAME을 선택하면 예측 결과 출력
    if soc_name_input:
        result = analyze_job(soc_name_input)
        st.write(f'### 예측된 비자 발급 결과입니다: {result}')

    if st.button('뒤로'):
        go_to_page('main')
