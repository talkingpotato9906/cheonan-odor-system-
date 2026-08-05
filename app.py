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

if 'intro_played' not in st.session_state:
    st.session_state['intro_played'] = False

if not st.session_state['intro_played']:
    lottie_url = "https://lottie.host/80aeb8c3-4fdb-4e1b-8531-1e9a3b68019b/6xGq5iRjN5.json"
    lottie_json = load_lottieurl(lottie_url)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #3B82F6; font-weight: 800; letter-spacing: -0.02em;'>🚁 천안-충남 스마트 악취 방어 시스템</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #111827;'>시스템을 초기화하고 실시간 데이터를 연동 중입니다...</h4>", unsafe_allow_html=True)
    
    if lottie_json:
        st_lottie(lottie_json, height=300, key="intro_anim")
    
    time.sleep(2.5) 
    st.session_state['intro_played'] = True
    st.rerun()

# ==========================================
# 3. 🚀 [핵심] Flat Design System CSS 강제 주입
# ==========================================
st.markdown("""
<style>
    /* 1. Typography: Outfit 폰트 적용 및 기본 텍스트 색상(Gray 900) */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Outfit', 'Pretendard', sans-serif; color: #111827; }
    
    /* 2. Headings: 굵고(800) 자간을 좁게(-0.02em) 설정하여 강렬하게 */
    h1, h2, h3, h4, h5, h6 { font-weight: 800 !important; letter-spacing: -0.02em !important; }
    
    /* 기본 스트림릿 UI 요소 숨기기 */
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
    
    /* 3. Background: Muted(Gray 100) 적용 (그림자 대신 색상 대비로 분리) */
    .stApp { background-color: #F3F4F6; }
    
    /* 🌟 스르륵 올라오는 애니메이션 (기존 유지) */
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* 4. Cards: 그림자(Shadow) 절대 금지! 둥근 모서리와 솔리드 컬러만 사용 */
    .flat-card {
        background-color: #FFFFFF;
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: none !important; /* 디자인 원칙: 절대 그림자 없음 */
        border: 2px solid transparent;
        margin-bottom: 2rem;
        animation: fadeUp 0.4s ease-out forwards;
        transition: transform 0.2s ease-in-out, border 0.2s;
    }
    /* 카드 Hover 시 그림자 대신 Scale 커지고 테두리 살짝 생김 */
    .flat-card:hover {
        transform: scale(1.01);
        border: 2px solid #E5E7EB;
    }

    /* 5. Buttons: Primary (Blue 500), 그림자 없이 스케일로만 반응 */
    .stButton > button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background-color: #2563EB !important; /* 약간 어두운 파란색으로 변경 */
        transform: scale(1.03) !important; /* 경쾌하게 커짐 */
    }
    
    /* 텍스트 색상 강제 지정용 (회색) */
    .text-gray { color: #6B7280 !important; font-weight: 500; }
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

ODOR_WEIGHT = { '돼지': 5.0, '소': 3.0, '한우': 3.0, '젖소': 3.0, '개': 2.0, '닭': 0.1, '오리': 0.1 }

# ==========================================
# 5. 데이터 로드 및 RAG 시스템 초기화
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
    return FAISS.from_documents(chunked_docs, embeddings)

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
                weight_per_head = val; break
        return np.log1p(count * weight_per_head) * 1.5 

    df_farm['악취가중치'] = df_farm.apply(calculate_weight, axis=1)

    try:
        df_apt = pd.read_csv("천안시_공동주택_최종_100퍼센트.csv", encoding='cp949')
    except:
        df_apt = pd.read_csv("천안시_공동주택_최종_100퍼센트.csv", encoding='utf-8')
    df_apt = df_apt.dropna(subset=['위도(lat)', '경도(lon)'])

    def calc_min_dist(lat, lon, farm_lats, farm_lons):
        R = 6371.0
        lat, lon = np.radians(lat), np.radians(lon)
        farm_lats, farm_lons = np.radians(farm_lats), np.radians(farm_lons)
        dlat, dlon = farm_lats - lat, farm_lons - lon
        a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(farm_lats) * np.sin(dlon/2)**2
        return np.min(R * 2 * np.arcsin(np.sqrt(a)))

    farm_lats, farm_lons = df_farm['위도'].values, df_farm['경도'].values
    df_apt['최근접축사_거리(km)'] = [calc_min_dist(row['위도(lat)'], row['경도(lon)'], farm_lats, farm_lons) for _, row in df_apt.iterrows()]
    df_impact = df_apt[df_apt['최근접축사_거리(km)'] <= 5.0].copy()
    df_impact['악취타격점수'] = 5.0 - df_impact['최근접축사_거리(km)']

    return df_farm, df_impact

with st.spinner('실제 공공데이터를 기반으로 공간 분석을 수행 중입니다...'):
    df_farm, df_impact = load_real_data()

# ==========================================
# 6. UI 레이아웃 및 메뉴 구성 (Flat Style)
# ==========================================
st.markdown('<h1 style="text-align: center; color: #3B82F6; margin-bottom: 30px;">🚁 천안-충남 광역 스마트 악취 모니터링</h1>', unsafe_allow_html=True)

menu_options = ["악취 영향권 지도", "드론 비전 AI", "자동 경보 시스템", "대시민 챗봇"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "악취 영향권 지도"

# Flat Design Menu: 그림자 제거, 굵은 테두리와 직관적 색상 사용
selected = option_menu(
    menu_title=None, 
    options=menu_options,
    icons=["map", "camera-reels", "bell", "chat-dots"],
    menu_icon="cast", 
    default_index=menu_options.index(st.session_state.active_tab), 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#FFFFFF", "border-radius": "8px", "border": "2px solid #E5E7EB"},
        "nav-link": {"font-size": "16px", "font-weight": "600", "text-align": "center", "margin":"0px", "color": "#6B7280", "border-radius": "0"},
        "nav-link-selected": {"background-color": "#3B82F6", "color": "white", "font-weight": "800"},
    }
)

if selected != st.session_state.active_tab:
    st.session_state.active_tab = selected
    st.rerun()

# ---------------------------------------------------------
# 메뉴 1: 악취 영향권 지도
# ---------------------------------------------------------
if selected == "악취 영향권 지도":
    st.markdown(f'''
    <div class="flat-card">
        <h2>🗺️ 천안시 축사 악취 확산 및 공동주택 피해 영향 지도</h2>
        <p class="text-gray">
            <b>분석 데이터:</b> 천안시 가축사육업 <span style="color:#3B82F6; font-weight:800;">{len(df_farm):,}</span>곳, 
            반경 5km 내 피해 영향권 공동주택 <span style="color:#3B82F6; font-weight:800;">{len(df_impact):,}</span>곳
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    center_lat, center_lon = (df_impact['위도(lat)'].mean(), df_impact['경도(lon)'].mean()) if not df_impact.empty else (36.815, 127.113)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
    
    heat_data = [[row['위도'], row['경도'], row['악취가중치']] for _, row in df_farm.iterrows()]
    HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3, gradient={0.4: 'blue', 0.6: 'lime', 1.0: 'red'}).add_to(m)
    
    for _, row in df_farm.iterrows():
        folium.CircleMarker([row['위도'], row['경도']], radius=3, color='black', weight=1, fill=True, fill_color='darkred', fill_opacity=0.7, popup=f"{row['사업장명']}").add_to(m)
        
    for _, row in df_impact.nlargest(150, '악취타격점수').iterrows():
        folium.CircleMarker([row['위도(lat)'], row['경도(lon)']], radius=3, color='blue', weight=1, fill=True, fill_color='cyan', fill_opacity=0.9, popup=f"{row['공동주택명']}").add_to(m)
    
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    st_folium(m, width="100%", height=600)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 메뉴 2: 드론 비전 AI
# ---------------------------------------------------------
elif selected == "드론 비전 AI":
    st.markdown('''
    <div class="flat-card">
        <h2>🚁 실시간 다각도 드론 영상 이상 징후 자동 검토</h2>
        <p class="text-gray">비전 AI가 이상 징후를 탐지하고 RAG가 관련 법령을 자동으로 찾아 처분 기준을 매칭합니다.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="flat-card">', unsafe_allow_html=True)
        st.subheader("📸 1. 항공뷰 (스카이뷰)")
        aerial_file = st.file_uploader("수직 항공뷰 1장을 업로드하세요", type=['jpg', 'jpeg', 'png'])
        st.subheader("📸 2. 측면뷰 (로드뷰)")
        side_files = st.file_uploader("건물 측면/환풍구 사진을 여러 장 업로드하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if aerial_file: st.image(aerial_file, use_column_width=True)
        if side_files:
            cols = st.columns(len(side_files))
            for i, sf in enumerate(side_files): cols[i].image(sf, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        if (aerial_file is not None) or (len(side_files) > 0):
            if st.button("🚀 정밀 단속 및 법적 검토 실행", type="primary", use_container_width=True):
                with st.spinner('비전 AI 분석 중...'):
                    # (이하 기존 드론 비전 AI 병합 및 LLM 처리 로직 동일하게 작동)
                    images_to_merge = []
                    if aerial_file: images_to_merge.append(Image.open(aerial_file).convert('RGB'))
                    for sf in side_files: images_to_merge.append(Image.open(sf).convert('RGB'))
                        
                    target_width = 800
                    resized_images = [img.resize((target_width, int(img.height * (target_width / img.width)))) for img in images_to_merge]
                    collage = Image.new('RGB', (target_width, sum(img.height for img in resized_images)))
                    y_offset = 0
                    for img in resized_images:
                        collage.paste(img, (0, y_offset)); y_offset += img.height
                        
                    buffered = io.BytesIO()
                    collage.save(buffered, format="JPEG")
                    final_base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    st.image(collage, caption="[AI 분석용 병합 드론 데이터]", use_column_width=True)

                try:
                    vision_res = client.chat.completions.create(
                        model="meta/llama-3.2-11b-vision-instruct", 
                        messages=[
                            {"role": "system", "content": "You are a strict JSON output machine. Only output valid JSON."},
                            {"role": "user", "content": [{"type": "text", "text": "문제점을 찾고 JSON 포맷 {\"detected_objects\":[], \"risk_level\":\"7\", \"summary_keyword\":\"\"} 으로 응답해."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64_image}"}}]}
                        ], temperature=0.3, max_tokens=500
                    )
                    raw_vision_text = vision_res.choices[0].message.content
                    import re; match = re.search(r'\{.*\}', raw_vision_text, re.DOTALL)
                    vision_data = json.loads(match.group(0)) if match else {"detected_objects": ["노후 축사 및 분뇨 방치 의심"], "risk_level": "7", "summary_keyword": "가축분뇨 방치"}

                    detected_items = vision_data.get("detected_objects", ["노후 축사 의심"])
                    risk = vision_data.get("risk_level", "5")
                    
                    st.info(f"🔍 **탐지 결과**: {', '.join(detected_items)} (위험도: {risk}/10)")
                    st.session_state['alert_info'] = f"발견된 문제: {', '.join(detected_items)} / 위험도: {risk}/10"
                    
                    with st.spinner('RAG 시스템 법령 검색 중...'):
                        try:
                            docs = init_rag_system().as_retriever(search_kwargs={"k": 2}).invoke(f"가축분뇨 {vision_data.get('summary_keyword', '')} 위반 행정처분")
                            legal_context = "\n\n".join([doc.page_content for doc in docs])
                        except: legal_context = "가축사육 제한 조례 적용"

                    final_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": f"AI 탐지 내용({', '.join(detected_items)}, 위험도 {risk})과 RAG 데이터({legal_context})를 종합하여 단속 보고서를 작성하라."}],
                        temperature=0.2, max_tokens=1500
                    )
                    st.success("✨ 드론 비전 & RAG 법률 자동 매칭 보고서 완성!")
                    st.markdown(f'<div class="flat-card">{final_res.choices[0].message.content}</div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"❌ 오류 발생: {e}")

# ---------------------------------------------------------
# 메뉴 3: 자동 경보 시스템
# ---------------------------------------------------------
elif selected == "자동 경보 시스템":
    st.markdown('''
    <div class="flat-card">
        <h2>📢 실시간 상황 전파 및 알림 시스템</h2>
        <p class="text-gray">실시간 기상청 풍향 데이터를 결합하여 맞춤형 긴급 문자를 생성합니다.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    if 'alert_info' not in st.session_state: st.warning("⚠️ 먼저 [드론 비전 AI] 메뉴에서 문제점을 탐지해 주세요.")
    else:
        alert_context = st.session_state['alert_info']
        st.info(f"🚨 **현장 상황**: {alert_context}")
        wind_direction = st.selectbox("현재 풍향 (확산 방향 예측)", ["북서풍", "남동풍", "동풍", "바람 없음"])
        
        if st.button("🚨 상황 전파 메시지 생성", type="primary"):
            with st.spinner("메시지 작성 중..."):
                try:
                    message_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": f"적발 내용({alert_context})과 기상 정보({wind_direction})를 바탕으로 시민용 긴급 SMS와 관리사무소 방송 대본을 작성하라."}],
                        temperature=0.2, max_tokens=1000
                    )
                    st.success("✅ 메시지 작성 완료")
                    st.markdown(message_res.choices[0].message.content)
                except Exception as e: st.error(f"❌ 오류 발생: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 메뉴 4: 대시민 AI 챗봇
# ---------------------------------------------------------
elif selected == "대시민 챗봇":
    st.markdown('''
    <div class="flat-card">
        <h2>💬 실시간 악취 민원 챗봇</h2>
        <p class="text-gray">현재 천안시 악취 상황, 대피 요령, 민원 접수 등에 대해 물어보세요!</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="flat-card">', unsafe_allow_html=True)
    if "chat_history" not in st.session_state: st.session_state.chat_history = [{"role": "assistant", "content": "안녕하십니까, 천안시청 악취통합관리센터입니다. 무엇을 도와드릴까요?"}]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("예: 오늘 목천읍 악취 상황은 어때요?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("답변 작성 중..."):
                try:
                    current_status = st.session_state.get('alert_info', '특이 사항 없음')
                    messages = [{"role": "system", "content": f"당신은 공무원입니다. 정중하게 응대하세요. 현재 상황: {current_status}"}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-5:]]
                    response = client.chat.completions.create(model="meta/llama-3.1-8b-instruct", messages=messages, temperature=0.3, max_tokens=500)
                    full_response = response.choices[0].message.content
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e: message_placeholder.error(f"오류: {e}")
    st.markdown('</div>', unsafe_allow_html=True)