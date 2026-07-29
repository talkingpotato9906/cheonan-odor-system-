import os
import io
import json
import glob
import base64
import time
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from pyproj import Transformer
from PIL import Image

import streamlit as st
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="천안시 스마트 악취 방어 시스템", page_icon="🚁", layout="wide")

# ==========================================
# 2. 멋진 인트로 화면 (Splash Screen) 로직
# ==========================================
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 세션을 이용해 접속 시 딱 1번만 인트로가 나오게 설정
if 'intro_played' not in st.session_state:
    st.session_state['intro_played'] = False

if not st.session_state['intro_played']:
    # 레이더/스캔 느낌의 Lottie 애니메이션
    lottie_url = "https://lottie.host/80aeb8c3-4fdb-4e1b-8531-1e9a3b68019b/6xGq5iRjN5.json"
    lottie_json = load_lottieurl(lottie_url)
    
    # 인트로 화면 중앙 정렬 및 렌더링
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🚁 천안-충남 스마트 악취 방어 시스템</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>시스템을 초기화하고 실시간 데이터를 연동 중입니다...</h4>", unsafe_allow_html=True)
    
    if lottie_json:
        st_lottie(lottie_json, height=300, key="intro_anim")
    
    time.sleep(2.5) # 2.5초간 인트로 감상 시간
    st.session_state['intro_played'] = True
    st.rerun() # 화면을 부드럽게 새로고침하여 메인 대시보드로 진입!

# ==========================================
# 3. 프리텐다드 폰트 및 🌟 탭 전환 애니메이션 추가
# ==========================================
st.markdown("""
<style>
    /* 웹 폰트(프리텐다드) 적용 */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: 'Pretendard', sans-serif;}
    
    /* 기본 스트림릿 UI 요소(헤더, 푸터, 메뉴) 숨기기 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 전체 배경을 아주 연한 회색으로 변경 (카드가 돋보이게) */
    .stApp {background-color: #F5F7FA;}
    
    /* 🌟 스르륵 올라오면서 켜지는 애니메이션 정의 */
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* 🌟 콘텐츠를 감싸는 예쁜 하얀색 카드 클래스 (애니메이션 강제 적용) */
    .modern-card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        animation: fadeUp 0.6s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. 환경 변수 및 API 설정
# ==========================================
load_dotenv()
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=120.0,
    max_retries=2
)

# --- 가축 종류별 악취 가중치 사전 정의 ---
ODOR_WEIGHT = {
    '돼지': 5.0,
    '소': 3.0,
    '한우': 3.0,
    '젖소': 3.0,
    '개': 2.0,
    '닭': 0.1,  
    '오리': 0.1
}

# ==========================================
# 5. 데이터 로드 및 RAG 시스템 초기화 (캐싱)
# ==========================================
@st.cache_resource
def init_rag_system(folder_path="rules"):
    documents = []
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    return vectorstore

@st.cache_data
def load_real_data():
    df_farm = pd.read_csv('천안시_가축사육업_정상영업_좌표완료_진짜최종.csv', encoding='utf-8')
    df_farm['사육두수'] = pd.to_numeric(df_farm['사육두수'], errors='coerce').fillna(0)
    df_farm['경도'] = pd.to_numeric(df_farm['경도'], errors='coerce')
    df_farm['위도'] = pd.to_numeric(df_farm['위도'], errors='coerce')
    df_farm = df_farm.dropna(subset=['위도', '경도'])

    def calculate_weight(row):
        species = str(row['주사육업종']).strip()
        count = row['사육두수']
        weight_per_head = 1.0
        for key, val in ODOR_WEIGHT.items():
            if key in species:
                weight_per_head = val
                break
        total_odor = count * weight_per_head
        return np.log1p(total_odor) * 1.5 

    df_farm['악취가중치'] = df_farm.apply(calculate_weight, axis=1)

    try:
        df_apt = pd.read_csv("천안시_공동주택_최종_100퍼센트.csv", encoding='cp949')
    except UnicodeDecodeError:
        df_apt = pd.read_csv("천안시_공동주택_최종_100퍼센트.csv", encoding='utf-8')
    df_apt = df_apt.dropna(subset=['위도(lat)', '경도(lon)'])

    def calc_min_dist(lat, lon, farm_lats, farm_lons):
        R = 6371.0
        lat, lon = np.radians(lat), np.radians(lon)
        farm_lats, farm_lons = np.radians(farm_lats), np.radians(farm_lons)
        dlat = farm_lats - lat
        dlon = farm_lons - lon
        a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(farm_lats) * np.sin(dlon/2)**2
        return np.min(R * 2 * np.arcsin(np.sqrt(a)))

    farm_lats = df_farm['위도'].values
    farm_lons = df_farm['경도'].values

    df_apt['최근접축사_거리(km)'] = [calc_min_dist(row['위도(lat)'], row['경도(lon)'], farm_lats, farm_lons) for _, row in df_apt.iterrows()]
    df_impact = df_apt[df_apt['최근접축사_거리(km)'] <= 5.0].copy()
    df_impact['악취타격점수'] = 5.0 - df_impact['최근접축사_거리(km)']

    return df_farm, df_impact

# 데이터 로딩 실행
with st.spinner('실제 공공데이터를 기반으로 공간 분석을 수행 중입니다...'):
    df_farm, df_impact = load_real_data()

# ==========================================
# 6. UI 레이아웃 및 메뉴 구성
# ==========================================
st.markdown('<h1 style="text-align: center; color: #1f77b4; margin-bottom: 20px;">🚁 천안-충남 광역 스마트 악취 통합 모니터링 플랫폼</h1>', unsafe_allow_html=True)

# --- 💡 탭 상태 저장을 위한 세션 추가 (버그 해결 핵심) ---
if "current_tab_idx" not in st.session_state:
    st.session_state.current_tab_idx = 0

menu_list = ["악취 영향권 지도", "드론 비전 AI", "자동 경보 시스템", "대시민 챗봇"]

# 탭 대신 세련된 상단 네비게이션 바 사용
selected = option_menu(
    menu_title=None, 
    options=menu_list,
    icons=["map", "camera-reels", "bell", "chat-dots"],
    menu_icon="cast", 
    default_index=st.session_state.current_tab_idx, # 고정된 0 대신 세션 기억장치 연결
    orientation="horizontal",
    key="main_menu", # 고유 식별 키 부여 (필수)
    styles={
        "container": {"padding": "0!important", "background-color": "#FFFFFF", "box-shadow": "0 2px 5px rgba(0,0,0,0.05)", "border-radius": "10px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "color": "#495057"},
        "nav-link-selected": {"background-color": "#1f77b4", "color": "white", "font-weight": "bold"},
    }
)

# 현재 선택된 탭의 번호를 세션에 업데이트하여 파란색 불빛이 튕기지 않게 고정
if selected:
    st.session_state.current_tab_idx = menu_list.index(selected)
    
    
# ---------------------------------------------------------
# 메뉴 1: 악취 영향권 지도
# ---------------------------------------------------------
if selected == "악취 영향권 지도":
    st.markdown(f'''
    <div class="modern-card">
        <h2>🗺️ 천안시 축사 악취 확산 및 공동주택 피해 영향 지도</h2>
        <p style="color:gray;">
            <b>분석 데이터:</b> 천안시 가축사육업 <span style="color:#1f77b4; font-weight:bold;">{len(df_farm):,}</span>곳, 
            반경 5km 내 피해 영향권 공동주택 <span style="color:#1f77b4; font-weight:bold;">{len(df_impact):,}</span>곳
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    if not df_impact.empty:
        center_lat = df_impact['위도(lat)'].mean()
        center_lon = df_impact['경도(lon)'].mean()
    else:
        center_lat, center_lon = 36.815, 127.113
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
    
    heat_data = [[row['위도'], row['경도'], row['악취가중치']] for _, row in df_farm.iterrows()]
    HeatMap(
        heat_data, radius=20, blur=15, min_opacity=0.3, 
        gradient={0.4: 'blue', 0.6: 'lime', 1.0: 'red'}
    ).add_to(m)
    
    for _, row in df_farm.iterrows():
        folium.CircleMarker(
            [row['위도'], row['경도']], radius=3, color='black', weight=1,
            fill=True, fill_color='darkred', fill_opacity=0.7, 
            popup=f"배출원: {row['사업장명']}<br>가축: {row['주사육업종']}<br>사육두수: {row['사육두수']}마리"
        ).add_to(m)
        
    for _, row in df_impact.nlargest(150, '악취타격점수').iterrows():
        folium.CircleMarker(
            [row['위도(lat)'], row['경도(lon)']], radius=3, color='blue', weight=1,
            fill=True, fill_color='cyan', fill_opacity=0.9,
            popup=f"{row['공동주택명']}<br>가장 가까운 축사: {row['최근접축사_거리(km)']:.2f}km"
        ).add_to(m)
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st_folium(m, width="100%", height=600)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📊 현재 지도에 반영된 데이터 통계 보기"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 모니터링 농가 수", f"{len(df_farm):,} 개소")
            st.dataframe(df_farm['주사육업종'].value_counts())
        with col2:
            st.metric("총 사육 두수", f"{int(df_farm['사육두수'].sum()):,} 두")
            st.dataframe(df_farm[['사업장명', '주사육업종', '사육두수', '악취가중치']].sort_values(by='악취가중치', ascending=False).head(10))

# ---------------------------------------------------------
# 메뉴 2: 드론 비전 AI
# ---------------------------------------------------------
elif selected == "드론 비전 AI":
    st.markdown('''
    <div class="modern-card">
        <h2>🚁 실시간 다각도 드론 영상 이상 징후 및 법률 자동 검토</h2>
        <p style="color:gray;">수직 항공뷰(스카이뷰)와 측면뷰를 업로드하면, <b>비전 AI가 이상 징후를 탐지하고 RAG가 관련 법령을 자동으로 찾아 처분 기준을 매칭</b>합니다.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("📸 1. 항공뷰 (스카이뷰)")
        aerial_file = st.file_uploader("수직 항공뷰 1장을 업로드하세요", type=['jpg', 'jpeg', 'png'], key="aerial_v2")
        
        st.subheader("📸 2. 측면뷰 (로드뷰)")
        side_files = st.file_uploader("건물 측면/환풍구 사진을 여러 장 업로드하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="side_v2")

        if aerial_file:
            st.image(aerial_file, caption="[항공뷰 미리보기]", use_column_width=True)
        if side_files:
            cols = st.columns(len(side_files))
            for i, sf in enumerate(side_files):
                cols[i].image(sf, caption=f"[측면뷰 {i+1}]", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        if (aerial_file is not None) or (len(side_files) > 0):
            if st.button("🚀 드론 정밀 단속 및 법적 검토 통합 실행", type="primary", use_container_width=True):
                with st.spinner('1단계: 다각도 이미지 병합 및 비전 AI 분석 중...'):
                    images_to_merge = []
                    if aerial_file:
                        images_to_merge.append(Image.open(aerial_file).convert('RGB'))
                    for sf in side_files:
                        images_to_merge.append(Image.open(sf).convert('RGB'))
                        
                    target_width = 800
                    resized_images = []
                    for img in images_to_merge:
                        ratio = target_width / img.width
                        new_height = int(img.height * ratio)
                        resized_images.append(img.resize((target_width, new_height)))
                        
                    total_height = sum(img.height for img in resized_images)
                    collage = Image.new('RGB', (target_width, total_height))
                    y_offset = 0
                    for img in resized_images:
                        collage.paste(img, (0, y_offset))
                        y_offset += img.height
                        
                    buffered = io.BytesIO()
                    collage.save(buffered, format="JPEG")
                    final_base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    st.image(collage, caption="[AI 분석용 병합 드론 데이터]", use_column_width=True)

                vision_prompt = """
                [지시사항]
                제공된 병합 이미지를 분석하여 문제점을 찾고, 반드시 아래의 JSON 형식으로만 응답하세요.
                다른 설명, 인사말, 마크다운(```json)은 절대 포함하지 마세요. 반드시 '{' 로 시작해야 합니다.
                
                {
                  "detected_objects": ["분뇨 야적장 방치", "축사 외벽 파손"],
                  "risk_level": "7",
                  "summary_keyword": "가축분뇨 유출 및 시설 파손"
                }
                """
                
                try:
                    vision_res = client.chat.completions.create(
                        model="meta/llama-3.2-11b-vision-instruct", 
                        messages=[
                            {"role": "system", "content": "You are a strict JSON output machine. Only output valid JSON."},
                            {"role": "user", "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64_image}"}}
                            ]}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    raw_vision_text = vision_res.choices[0].message.content
                    
                    import re
                    match = re.search(r'\{.*\}', raw_vision_text, re.DOTALL)
                    
                    if match:
                        try:
                            vision_data = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            vision_data = {
                                "detected_objects": ["시각적 분석 오류 (수동 확인 필요)"],
                                "risk_level": "5",
                                "summary_keyword": "악취 배출 시설"
                            }
                    else:
                        vision_data = {
                            "detected_objects": ["노후 축사 및 분뇨 방치 의심"],
                            "risk_level": "7",
                            "summary_keyword": "가축분뇨 방치"
                        }

                    detected_items = vision_data.get("detected_objects", ["노후 축사 의심"])
                    search_keyword = vision_data.get("summary_keyword", "가축분뇨 유출 및 시설 파손")
                    risk = vision_data.get("risk_level", "5")
                    
                    st.info(f"🔍 **비전 AI 탐지 결과**: {', '.join(detected_items)} (위험도: {risk}/10)")
                    st.session_state['alert_info'] = f"발견된 문제: {', '.join(detected_items)} / 위험도: {risk}/10"
                    
                    with st.spinner('2단계: RAG 시스템이 관련 법령을 자동 검색 및 매칭 중...'):
                        try:
                            vectorstore = init_rag_system()
                            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                            docs = retriever.invoke(f"가축분뇨 {search_keyword} 위반 행정처분 과태료 조례")
                            legal_context = "\n\n".join([doc.page_content for doc in docs])
                        except Exception as e:
                            legal_context = "천안시 가축사육 제한 조례 및 가축분뇨의 관리 및 이용에 관한 법률 적용"

                    final_report_prompt = f"""
                    당신은 '환경부 소속 가축분뇨 악취 단속 수석 조사관'입니다.
                    아래의 [비전 AI 탐지 내용]과 [RAG 법령 데이터]를 종합하여, 현장 출동 및 행정 처분에 바로 사용할 수 있는 '공식 단속 결과 보고서'를 작성하세요.
                    
                    [비전 AI 탐지 내용]
                    - 발견된 문제: {", ".join(detected_items)}
                    - 위험도 평가: {risk}/10
                    
                    [참고 법령 데이터 (RAG)]
                    {legal_context}
                    
                    [보고서 작성 지침]
                    1. 개요: 드론 모니터링을 통한 현장 적발 내용
                    2. 위반 사항 분석: 포착된 취약 요소 상세 지적
                    3. 적용 법령 및 행정 조치: RAG 데이터를 바탕으로 구체적인 법조항과 예상 과태료/조치명령 명시
                    4. 조치 의견: 농장주 시정 요구 사항
                    """

                    final_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": final_report_prompt}],
                        temperature=0.2,
                        max_tokens=1500
                    )
                    
                    st.success("✨ 드론 비전 & RAG 법률 자동 매칭 보고서 완성!")
                    st.markdown(f'<div class="modern-card">{final_res.choices[0].message.content}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ 분석 중 오류 발생: {e}")

# ---------------------------------------------------------
# 메뉴 3: 자동 경보 시스템
# ---------------------------------------------------------
elif selected == "자동 경보 시스템":
    st.markdown('''
    <div class="modern-card">
        <h2>📢 실시간 상황 전파 및 알림 시스템</h2>
        <p style="color:gray;">단속 결과와 <b>실시간 기상청 풍향 데이터</b>를 결합하여, 악취 확산이 예상되는 타겟 지역의 시민들에게만 맞춤형 긴급 문자 및 방송 대본을 생성합니다.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    if 'alert_info' not in st.session_state:
        st.warning("⚠️ 먼저 [드론 비전 AI] 메뉴에서 드론 정밀 단속을 실행하여 문제점을 탐지해 주세요.")
    else:
        alert_context = st.session_state['alert_info']
        st.info(f"🚨 **전달받은 현장 상황**: {alert_context}")
        
        st.subheader("🌤️ 실시간 기상 데이터 연동")
        wind_direction = st.selectbox(
            "현재 풍향을 선택하세요 (악취 확산 방향 예측)",
            [
                "북서풍 (남동쪽 대규모 주거단지 방향으로 악취 확산 예상)",
                "남동풍 (북서쪽 초등학교 및 상가 밀집 지역으로 악취 확산 예상)",
                "동풍 (서쪽 호수공원 및 산책로 방향으로 악취 확산 예상)",
                "바람 없음 (농장 인근 반경 1km 이내 전 지역 악취 체류 중)"
            ]
        )
        
        if st.button("🚨 풍향 맞춤형 상황 전파 메시지 생성", type="primary"):
            with st.spinner("풍향 데이터를 분석하여 피해 예상 지역 주민들을 위한 메시지를 작성 중입니다..."):
                message_prompt = f"""
                당신은 '천안시 재난안전상황실의 최고 시민 소통 담당관'입니다.
                아래의 [현장 적발 내용]과 [실시간 기상 정보]를 종합하여, 악취 피해가 직접적으로 예상되는 지역 주민들을 위한 두 가지 전파 메시지를 작성하세요.
                
                [현장 적발 내용]
                {alert_context}
                
                [실시간 기상(풍향) 정보]
                {wind_direction}
                
                [작성 지침]
                1. 시민 발송용 긴급 SMS (150자 이내): 악취 발생 사실, '풍향에 따른 실제 피해 예상 구역' 명확히 지목, 창문 닫기 및 야외활동 자제 등 행동 요령 포함. 간결하고 신속한 재난 문자 톤.
                2. 관리사무소/마을회관 방송 대본: 풍향 데이터에 따라 피해가 예상되는 아파트 단지나 마을 주민들에게 상황을 구체적으로 설명하고 협조를 구하는 차분하고 친절한 구어체 대본.
                
                반드시 아래의 형식에 맞춰서 출력해 주세요. 마크다운 외에 다른 불필요한 인사말은 생략하세요.
                
                ### 📱 타겟 시민 발송용 긴급 문자 (SMS)
                (문자 내용)
                
                ---
                
                ### 🎙️ 확산 예상 지역 관리사무소 방송 대본
                (대본 내용)
                """
                
                try:
                    message_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": message_prompt}],
                        temperature=0.2, 
                        max_tokens=1000
                    )
                    
                    st.success("✅ 실시간 풍향이 반영된 타겟형 전파 메시지 작성이 완료되었습니다.")
                    st.markdown(message_res.choices[0].message.content)
                    
                    st.divider()
                    st.markdown("#### 🚀 원클릭 전송 시스템 (Demo)")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("✉️ 악취 확산 예상 지역 시민들에게만 문자 발송", use_container_width=True):
                            wind_type = wind_direction.split(" ")[0]
                            st.toast(f"[{wind_type}] 영향을 받는 타겟 구역 시민들에게 긴급 문자가 발송되었습니다!", icon="📱")
                    with cols[1]:
                        if st.button("📞 해당 방향 유관기관 및 관리사무소 대본 자동 전송", use_container_width=True):
                            st.toast("피해 예상 지역의 아파트 관리사무소 12곳에 대본 전송이 완료되었습니다!", icon="🎙️")
                            
                except Exception as e:
                    st.error(f"❌ 메시지 생성 중 오류가 발생했습니다: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 메뉴 4: 대시민 AI 챗봇
# ---------------------------------------------------------
elif selected == "대시민 챗봇":
    st.markdown('''
    <div class="modern-card">
        <h2>💬 실시간 악취 민원 챗봇</h2>
        <p style="color:gray;">현재 천안시 악취 상황, 대피 요령, 민원 접수 등에 대해 자유롭게 물어보세요!</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "안녕하십니까, 천안시청 환경과 악취통합관리센터입니다. 악취 관련 민원이나 궁금한 사항을 말씀해 주시면 신속하게 안내해 드리겠습니다."}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("예: 오늘 목천읍 주변 악취 상황은 어때요? 조치 중인가요?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("민원 내용을 확인하고 답변을 작성 중입니다..."):
                current_status = st.session_state.get('alert_info', '현재 드론 모니터링 시스템상 특이 사항이 보고되지 않았습니다. 지속적인 모니터링을 실시하겠습니다.')
                
                system_prompt = f"""
                당신은 천안시청 환경과 악취통합관리센터 소속 주무관(공무원)을 대신하여 민원을 응대하는 AI 시스템입니다.
                답변은 반드시 실제 공무원이 시민의 민원을 응대하는 정중하고 격식 있는 공공기관 행정 톤앤매너를 엄격하게 지켜주세요.
                
                [응대 가이드라인]
                1. 어조: '~습니다', '~안내해 드립니다', '~하시기 바랍니다', '양해 부탁드립니다' 등 다/나/까 기반의 정중한 경어체 사용.
                2. 태도: 시민의 불편에 먼저 정중히 공감 및 사과하되, 감정적으로 흔들리지 않고 객관적이고 단호한 매뉴얼 기반 안내.
                3. 용어: '조치 중입니다', '현장 점검을 실시하겠습니다', '행정 처분을 검토 중입니다' 등 실제 행정 용어 적극 활용.
                
                현재 모니터링 시스템에 접수된 상황은 다음과 같습니다: 
                {current_status}
                
                위 정보를 바탕으로 시민의 질문에 대해 공공기관의 공식 답변서처럼 체계적이고 신뢰감 있게 안내해 주세요.
                """
                
                messages = [{"role": "system", "content": system_prompt}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[-5:]
                ]

                try:
                    response = client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=500
                    )
                    full_response = response.choices[0].message.content
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    message_placeholder.error(f"응답 생성 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주시기 바랍니다. (상세 오류: {e})")
    st.markdown('</div>', unsafe_allow_html=True)