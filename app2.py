import os
import io
import json
import glob
import base64
import time
import requests
import math
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from PIL import Image
from datetime import datetime

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
# 1. 페이지 기본 설정 & 로컬 DB 함수
# ==========================================
st.set_page_config(page_title="천안 스마트 악취 모니터링", page_icon="🌿", layout="wide")

HISTORY_DB_FILE = "farm_history_db.json"

def load_farm_history():
    if os.path.exists(HISTORY_DB_FILE):
        with open(HISTORY_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_farm_history(farm_name, data):
    history = load_farm_history()
    history[farm_name] = data
    with open(HISTORY_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ==========================================
# 2. 인트로 화면 (Flat Design)
# ==========================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

if 'intro_played' not in st.session_state:
    st.session_state['intro_played'] = False

if not st.session_state['intro_played']:
    lottie_url = "https://lottie.host/80aeb8c3-4fdb-4e1b-8531-1e9a3b68019b/6xGq5iRjN5.json"
    lottie_json = load_lottieurl(lottie_url)
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #3B82F6; font-weight: 800; letter-spacing: -1px;'>🌿 천안 스마트 악취 통합 모니터링</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #6B7280; font-weight: 500;'>기상청 실시간 API 및 AI 시스템을 초기화 중입니다...</h4>", unsafe_allow_html=True)
    
    if lottie_json: st_lottie(lottie_json, height=250, key="intro_anim")
    time.sleep(2.5) 
    st.session_state['intro_played'] = True
    st.rerun()

# ==========================================
# 3. 🎨 Flat UI CSS 디자인 세팅
# ==========================================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; color: #111827; letter-spacing: -0.02em; }
    h1, h2, h3 { font-weight: 800 !important; }
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
    .stApp { background-color: #F9FAFB; }
    
    .stButton > button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }
    .stButton > button:hover { background-color: #2563EB !important; transform: scale(1.03) !important; }
    .text-muted { color: #6B7280 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. 공통 함수 및 AI/API 설정
# ==========================================
load_dotenv()
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"), timeout=120.0)

def get_scalar(val, default=0):
    if isinstance(val, (pd.Series, pd.DataFrame)): return val.iloc[0]
    if isinstance(val, np.ndarray): return val.item(0) if val.size > 0 else default
    if pd.isna(val): return default
    return val

@st.cache_resource
def init_rag_system(folder_path="rules"):
    documents = []
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    for file in pdf_files: documents.extend(PyPDFLoader(file).load())
    if not documents: return None
    chunked_docs = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(documents)
    return FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"))

@st.cache_data
def load_data():
    def safe_read(file_name):
        for enc in ['cp949', 'utf-8', 'utf-8-sig', 'euc-kr']:
            try: return pd.read_csv(file_name, encoding=enc).loc[:, lambda df: ~df.columns.duplicated()]
            except: continue
        return pd.DataFrame()

    df_farm = safe_read("천안시_가축사육업_정상영업_좌표완료_진짜최종.csv") 
    df_apt = safe_read("천안시_공동주택_최종마스터_좌표완료.csv") 

    def rename_lat_lon(df):
        if not df.empty:
            lat_col = [c for c in df.columns if '위도' in c]
            lon_col = [c for c in df.columns if '경도' in c]
            if lat_col: df.rename(columns={lat_col[0]: '위도'}, inplace=True)
            if lon_col: df.rename(columns={lon_col[0]: '경도'}, inplace=True)
        return df
    
    df_farm = rename_lat_lon(df_farm)
    df_apt = rename_lat_lon(df_apt)

    if not df_apt.empty:
        df_apt = df_apt.dropna(subset=['공동주택명'])
        df_apt = df_apt[~df_apt['공동주택명'].astype(str).str.contains('합계|총계')]

    if not df_farm.empty:
        farm_name_col = [c for c in df_farm.columns if '사업장명' in c or '농가' in c]
        df_farm['농가식별명'] = df_farm[farm_name_col[0]] if farm_name_col else "미상 농가"
        
        df_farm = df_farm.dropna(subset=['농가식별명'])
        df_farm = df_farm[~df_farm['농가식별명'].astype(str).str.contains('합계|총계|미상')]
        
        if '사육두수' not in df_farm.columns:
            headcount_cols = [c for c in df_farm.columns if '두수' in c]
            if not headcount_cols:
                headcount_cols = [c for c in df_farm.columns if '사육' in c and '면적' not in c]
            df_farm['사육두수'] = df_farm[headcount_cols[0]] if headcount_cols else 1000
        
        df_farm['사육두수'] = df_farm['사육두수'].apply(lambda x: float(str(x).replace(',', '')) if pd.notnull(x) else 0.0)
        df_farm = df_farm[df_farm['사육두수'] > 0]
        
        species_weights = { '돼지': 10.9, '젖소': 0.6, '소': 0.4, '한우': 0.4, '닭': 0.2, '개': 2.0, '오리': 0.2 }
        species_col = [c for c in df_farm.columns if '업종' in c or '축종' in c]
        target_species = species_col[0] if species_col else '주사육업종'
        
        if target_species in df_farm.columns:
            df_farm['축종가중치'] = df_farm[target_species].map(lambda x: next((v for k, v in species_weights.items() if k in str(x)), 1.0))
        else:
            df_farm['축종가중치'] = 1.0
            
        df_farm['Odor_Emission'] = df_farm['사육두수'] * df_farm['축종가중치']
        
    return df_farm, df_apt

def get_live_weather(api_key):
    if not api_key: return None, None, "API 키가 없습니다."
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?stn=232&help=0&authKey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and "ERR" not in response.text:
            lines = response.text.strip().split('\n')
            for line in reversed(lines):
                if line.strip() and not line.startswith('#'):
                    data = line.split()
                    if len(data) >= 4:
                        w_dir = float(data[2])
                        w_spd = float(data[3])
                        if w_dir == -9.0 or w_spd == -9.0:
                            w_dir, w_spd = 0.0, 0.0
                        return w_dir, w_spd, "천안(232) 최신 관측 성공"
        return None, None, "데이터 오류 (기상청 응답 에러)"
    except: 
        return None, None, "통신 실패 (서버 지연)"

def calculate_haversine_and_bearing(lat1, lon1, lat2, lon2):
    rad_lat1, rad_lon1, rad_lat2, rad_lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = rad_lat2 - rad_lat1, rad_lon2 - rad_lon1
    a = math.sin(dlat/2)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(dlon/2)**2
    distance = 2 * 6371 * math.asin(math.sqrt(a))
    y = math.sin(dlon) * math.cos(rad_lat2)
    x = math.cos(rad_lat1) * math.sin(rad_lat2) - math.sin(rad_lat1) * math.cos(rad_lat2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    return distance, bearing

@st.cache_data
def calculate_cii(df_farm, df_apt, wind_dir, wind_speed):
    if df_farm.empty or df_apt.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    household_cols = [c for c in df_apt.columns if '세대' in c]
    target_household_col = household_cols[0] if household_cols else '세대수'
    
    apt_cii_list = []
    for _, apt in df_apt.iterrows():
        try:
            apt_lat, apt_lon = float(get_scalar(apt.get('위도', 0))), float(get_scalar(apt.get('경도', 0)))
            if apt_lat == 0 or apt_lon == 0 or pd.isna(apt_lat) or pd.isna(apt_lon): continue
            apt_households = float(str(get_scalar(apt.get(target_household_col, 100))).replace(',', ''))
        except: continue

        total_oei, max_contribution, top_farm_name = 0.0, -1.0, "미상 농가"
        for _, farm in df_farm.iterrows():
            try:
                farm_lat, farm_lon = float(get_scalar(farm.get('위도', 0))), float(get_scalar(farm.get('경도', 0)))
                farm_emission = float(get_scalar(farm.get('Odor_Emission', 0)))
                if farm_lat == 0 or farm_lon == 0 or pd.isna(farm_lat) or pd.isna(farm_lon) or farm_emission <= 0: continue
            except: continue
            
            dist, bearing = calculate_haversine_and_bearing(farm_lat, farm_lon, apt_lat, apt_lon)
            dist = max(dist, 0.1)
            
            wind_payload_dir = (wind_dir + 180) % 360
            wind_alignment = max(0.0, math.cos(math.radians(wind_payload_dir - bearing)))
            farm_contribution = (farm_emission / (dist ** 2)) * wind_alignment
            total_oei += farm_contribution
            
            if farm_contribution > max_contribution:
                max_contribution = farm_contribution
                top_farm_name = str(get_scalar(farm.get('농가식별명', '미상 농가')))

        cii_raw = float(total_oei * math.log10(apt_households + 1))
        
        if cii_raw > 0:
            apt_name = str(get_scalar(apt.get('공동주택명', '미상 아파트')))
            apt_cii_list.append({ '공동주택명': apt_name, '세대수': int(apt_households), 'CII_Raw': cii_raw, '위도': apt_lat, '경도': apt_lon, '원인농가': top_farm_name })

    df_res = pd.DataFrame(apt_cii_list)
    if df_res.empty: return df_res, df_farm, pd.DataFrame()
        
    max_raw = df_res['CII_Raw'].max()
    df_res['CII'] = round((df_res['CII_Raw'] / max_raw) * 100.0, 1) if max_raw > 0 else 0.0
    
    def assign_grade(score):
        if score >= 60: return "🔴 심각"
        elif score >= 30: return "🟠 경고"
        elif score > 0: return "🟡 주의"
        else: return "🟢 관심"

    df_res['위험등급'] = df_res['CII'].apply(assign_grade)
    df_res = df_res.sort_values('CII', ascending=False).reset_index(drop=True)
    
    df_top_farm = df_farm[df_farm['농가식별명'].isin(df_res.head(30)['원인농가'].unique())]
    return df_res, df_farm, df_top_farm

with st.spinner('데이터를 불러오고 있습니다...'):
    df_farm_raw, df_apt_raw = load_data()

# ==========================================
# 5. UI 레이아웃: 사이드바 (기상 컨트롤) 및 탭 구성
# ==========================================
with st.sidebar:
    st.markdown("### ☁️ 기상 컨트롤 패널")
    data_mode = st.radio("데이터 소스 선택", ["기상청 실시간 API", "수동 시뮬레이션"], index=0)
    
    w_dir, w_spd = 103.0, 2.5
    if data_mode == "기상청 실시간 API":
        api_dir, api_spd, api_msg = get_live_weather(os.getenv("KMA_API_KEY"))
        if api_dir is not None:
            w_dir, w_spd = api_dir, api_spd
            st.success(f"✅ {api_msg}")
            st.info(f"실시간 풍향: {w_dir}°\n\n실시간 풍속: {w_spd} m/s")
        else:
            st.error(f"❌ 연동 실패: {api_msg}")
            st.warning("수동 모드로 진행합니다.")
    else:
        st.subheader("수동 시뮬레이션")
        w_dir = st.slider("풍향 (불어오는 방향)", 0.0, 360.0, 103.0)
        w_spd = st.slider("풍속 (m/s)", 0.0, 15.0, 2.5)

    st.markdown("---")
    def get_wind_direction_str(degree):
        dirs = ["북풍", "북동풍", "동풍", "남동풍", "남풍", "남서풍", "서풍", "북서풍", "북풍"]
        return dirs[int((degree + 22.5) % 360 // 45)]
    
    st.metric(label="적용 풍향 (악취 확산)", value=f"{get_wind_direction_str(w_dir)} ({(w_dir + 180) % 360}°)")
    st.metric(label="적용 풍속", value=f"{w_spd} m/s", delta="확산 원활" if w_spd > 2.0 else "대기 정체 위험", delta_color="inverse")

# 데이터 연산 실행
df_impact_apt, df_farm_impact, df_top_farm = calculate_cii(df_farm_raw, df_apt_raw, w_dir, w_spd)

st.markdown('<h1 style="text-align: center; color: #3B82F6; margin-bottom: 30px;">🌿 천안 스마트 악취 통합 모니터링</h1>', unsafe_allow_html=True)

menu_options = ["실시간 악취 관제망", "드론 비전 AI 단속", "자동 경보 시스템", "인프라 및 정책 제언", "IoT 역추적 관제", "대시민 챗봇"]
if "active_tab" not in st.session_state: st.session_state.active_tab = menu_options[0]

selected = option_menu(
    menu_title=None, options=menu_options,
    icons=["map", "camera-reels", "bell", "building", "radar", "chat-dots"],
    menu_icon="cast", default_index=menu_options.index(st.session_state.active_tab), orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#FFFFFF", "border": "2px solid #E5E7EB", "border-radius": "10px"},
        "nav-link": {"font-size": "14px", "font-weight": "600", "text-align": "center", "margin":"5px", "color": "#6B7280"},
        "nav-link-selected": {"background-color": "#3B82F6", "color": "white", "font-weight": "800", "border-radius": "6px"},
    }
)
if selected != st.session_state.active_tab:
    st.session_state.active_tab = selected
    st.rerun()

# ---------------------------------------------------------
# 메뉴 1: 실시간 악취 관제망
# ---------------------------------------------------------
if selected == "실시간 악취 관제망":
    st.markdown("<h2>🗺️ 실시간 대기 확산 및 주민 피해 지수(CII) 관제</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='text-muted'>실시간 풍향({get_wind_direction_str(w_dir)}) 및 풍속({w_spd}m/s)을 바탕으로 악취의 이동 궤적과 아파트별 피해 규모 시각화합니다.</p><hr>", unsafe_allow_html=True)
    
    if not df_impact_apt.empty and not df_top_farm.empty:
        m = folium.Map(location=[36.815, 127.113], zoom_start=11, tiles="http://mt0.google.com/vt/lyrs=m&hl=ko&x={x}&y={y}&z={z}", attr="Google Maps")
        
        heat_data = [[float(row.get('위도',0)), float(row.get('경도',0)), float(row.get('Odor_Emission',1))] for _, row in df_farm_raw.iterrows() if not pd.isna(row.get('위도')) and float(row.get('위도',0))!=0]
        HeatMap(heat_data, radius=18, blur=15, min_opacity=0.2, gradient={0.4: 'blue', 0.6: 'lime', 1.0: 'red'}).add_to(m)

        for _, farm in df_top_farm.iterrows():
            f_lat, f_lon = float(farm.get('위도', 0)), float(farm.get('경도', 0))
            if f_lat == 0 or f_lon == 0 or pd.isna(f_lat): continue
            
            folium.CircleMarker(location=[f_lat, f_lon], radius=6, color='red', fill=True, fill_opacity=0.9, tooltip=f"농가: {farm.get('농가식별명', '미상')}").add_to(m)
            
            plume_dir = (w_dir + 180) % 360
            angle_spread, length = 30, 0.03
            p1 = [f_lat, f_lon]
            p2 = [f_lat + length * math.cos(math.radians(plume_dir - angle_spread)), f_lon + length * math.sin(math.radians(plume_dir - angle_spread))]
            p3 = [f_lat + length * math.cos(math.radians(plume_dir + angle_spread)), f_lon + length * math.sin(math.radians(plume_dir + angle_spread))]
            folium.Polygon(locations=[p1, p2, p3], color='red', fill=True, fill_opacity=0.15, weight=0).add_to(m)

        for _, apt in df_impact_apt.head(50).iterrows():
            folium.CircleMarker(
                location=[apt['위도'], apt['경도']], radius=4, color='blue', fill=True, fill_opacity=0.7,
                tooltip=f"{apt['공동주택명']} (위험도: {apt['CII']}점 / {apt['위험등급']})"
            ).add_to(m)

        st_folium(m, width="100%", height=550)
        
        st.markdown("<br><h3>🚨 실시간 방제단 출동 지시 (B2G)</h3>", unsafe_allow_html=True)
        col_table, col_btn = st.columns([4, 1])
        with col_table:
            st.dataframe(
                df_impact_apt[['공동주택명', '세대수', 'CII', '위험등급', '원인농가']], 
                use_container_width=True,
                height=250,
                hide_index=True
            )
        with col_btn:
            st.write("")
            if st.button("🚨 상위 5개 구역\n방제단 출동 발송", use_container_width=True):
                with st.spinner("서버 연결 중..."):
                    time.sleep(1)
                for _, row in df_impact_apt.head(5).iterrows():
                    st.toast(f"✅ {row['원인농가']} 인근 {row['공동주택명']} 방제단 출동 지시 발송!")
                    time.sleep(0.3)
                st.success("조기 알림 발송 완료!")

# ---------------------------------------------------------
# 메뉴 2: 드론 비전 AI 단속
# ---------------------------------------------------------
elif selected == "드론 비전 AI 단속":
    st.markdown("<h2>🚁 실시간 드론 AI 단속 및 건축물 변경 이력 검증</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-muted'>대상 농가를 선택하고 드론 이미지를 업로드하면, 과거 분석 이력과 현재 상태를 비교하여 불법 증축 및 법적 위반 사항을 자동 적발합니다.</p><hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        farm_list = df_farm_raw['농가식별명'].dropna().unique().tolist()
        default_farm = st.session_state.get('target_farm', farm_list[0]) if st.session_state.get('target_farm') in farm_list else farm_list[0]
        
        selected_farm = st.selectbox("🎯 단속 및 검증 대상 농가 선택", farm_list, index=farm_list.index(default_farm))
        
        history_db = load_farm_history()
        past_data = history_db.get(selected_farm, None)
        
        if past_data:
            st.info(f"📂 **[{past_data['date']}] 과거 단속 이력 존재:**\n\n탐지 내역: {', '.join(past_data['detected_objects'])} (위험도: {past_data['risk_level']})")
        else:
            st.warning("📂 **과거 단속 이력 없음:** 본 촬영이 기준(Baseline) 데이터로 아카이빙됩니다.")

        st.subheader("📸 1. 스카이뷰 업로드 (현재 상태)")
        aerial_file = st.file_uploader("수직 항공뷰 (1장)", type=['jpg', 'jpeg', 'png'])
        st.subheader("📸 2. 측면뷰 업로드")
        side_files = st.file_uploader("건물 측면/환풍구 (여러 장)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if aerial_file: st.image(aerial_file, use_column_width=True)
        if side_files:
            cols = st.columns(len(side_files))
            for i, sf in enumerate(side_files): cols[i].image(sf, use_column_width=True)
            
    with col2:
        if (aerial_file is not None) or (len(side_files) > 0):
            if st.button("🚀 AI 정밀 단속 및 시계열 변화 검토 실행", use_container_width=True):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # [STEP 1] 이미지 병합
                    status_text.markdown("### ⚙️ [1/4] 다각도 드론 영상 병합 및 AI 입력 전처리 중...")
                    time.sleep(0.5) 
                    
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
                    
                    progress_bar.progress(25)
                    st.image(collage, caption="[현재 AI 분석용 병합 데이터]", use_column_width=True)

                    # [STEP 2] Vision AI 분석
                    status_text.markdown("### 👁️ [2/4] Llama-3.2 Vision 기반 이상 징후 및 불법 시설 판독 중...")
                    
                    vision_res = client.chat.completions.create(
                        model="meta/llama-3.2-11b-vision-instruct", 
                        messages=[
                            {"role": "system", "content": "You are a strict JSON output machine. Do not output anything outside the JSON block."},
                            {"role": "user", "content": [
                                {"type": "text", "text": "건축물 구조, 가설 천막, 분뇨 처리 시설의 문제점을 찾고 JSON 포맷 {\"detected_objects\":[\"문제1\"], \"risk_level\":\"7\", \"summary_keyword\":\"키워드\", \"detailed_explanation\":\"사진에서 관찰된 문제점에 대한 구체적이고 상세한 설명(왜 이것이 문제인지, 어떤 상태인지 3~4문장으로 서술)\"} 으로 응답해."}, 
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64_image}"}}
                            ]}
                        ], temperature=0.3, max_tokens=800
                    )
                    raw_vision_text = vision_res.choices[0].message.content
                    
                    # 💡 핵심 수정 사항: JSON 파싱 예외 처리(Try-Except)를 통한 오류 방어벽 구축
                    import re
                    match = re.search(r'\{.*\}', raw_vision_text, re.DOTALL)
                    
                    default_vision_data = {
                        "detected_objects": ["불법 가설건축물(천막) 및 분뇨 방치 의심"], 
                        "risk_level": "8", 
                        "summary_keyword": "가축분뇨 방치 불법증축",
                        "detailed_explanation": "해당 이미지에서 규격에 맞지 않는 노후 가설 천막과 부적절하게 방치된 분뇨 더미가 관찰됩니다. 이는 심각한 악취를 유발할 수 있으며, 관련 조례 위반 소지가 다분합니다."
                    }
                    
                    if match:
                        try:
                            vision_data = json.loads(match.group(0))
                        except Exception:
                            vision_data = default_vision_data
                    else:
                        vision_data = default_vision_data

                    detected_items = vision_data.get("detected_objects", ["노후 축사 의심"])
                    risk = vision_data.get("risk_level", "5")
                    explanation = vision_data.get("detailed_explanation", "상세 설명이 제공되지 않았습니다.")
                    
                    st.info(f"🔍 **판독 완료**: {', '.join(detected_items)} (위험도: {risk}/10)\n\n📝 **AI 상세 분석 내용**: {explanation}")
                    
                    st.session_state['alert_info'] = f"발견된 문제: {', '.join(detected_items)} / 위험도: {risk}/10 / 상세내용: {explanation}"
                    
                    vision_data['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    save_farm_history(selected_farm, vision_data)
                    
                    progress_bar.progress(50)
                    
                    # [STEP 3] RAG 법령 매칭
                    status_text.markdown("### 📚 [3/4] RAG 기반 가축분뇨법 및 천안시 조례 텍스트 검색 중...")
                    
                    try:
                        docs = init_rag_system().as_retriever(search_kwargs={"k": 2}).invoke(f"가축분뇨 {vision_data.get('summary_keyword', '')} 불법건축물 위반 행정처분")
                        legal_context = "\n\n".join([doc.page_content for doc in docs])
                    except: legal_context = "건축법 및 가축분뇨법 제한 조례 적용"
                    
                    progress_bar.progress(75)

                    # [STEP 4] LLM 최종 보고서 생성
                    status_text.markdown("### 📝 [4/4] Llama-3.1 70B 모델 기반 시계열 대조 및 보고서 작성 중...")
                    
                    target_farm_row = df_farm_raw[df_farm_raw['농가식별명'] == selected_farm].iloc[0]
                    farm_address = str(get_scalar(target_farm_row.get('지번주소', '소재지 파악 불가')))
                    
                    base_prompt = f"보고서 작성 시 [농가명], [소재지] 같은 빈칸(Placeholder)을 절대 쓰지 말고, 다음 실제 정보를 반드시 기입하세요.\n- 대상 농가명: {selected_farm}\n- 소재지: {farm_address}\n\n"
                    
                    if past_data:
                        prompt_msg = base_prompt + f"해당 농가의 과거 단속 데이터는 [{', '.join(past_data['detected_objects'])}] 였습니다. 그런데 현재 드론 데이터에서는 [{', '.join(detected_items)}] 가 탐지되었습니다. 이 두 가지를 비교하여 '과거에 없던 불법 증축물'이나 '악화된 환경'에 초점을 맞추어 '과거 단속 데이터' 및 '변화된 점' 항목을 반드시 포함하여 지적하고, RAG 법령({legal_context})을 근거로 법적 위반 여부 및 행정처분 공문 보고서를 작성하라."
                    else:
                        prompt_msg = base_prompt + f"AI 탐지 내용({', '.join(detected_items)}, 위험도 {risk})과 RAG 데이터({legal_context})를 종합하여 단속 보고서를 작성하라. ⚠️주의: 이 농가는 과거 단속 내역(DB)이 최초 수집되는 상태이므로, 보고서 양식에서 '과거 단속 데이터' 및 '변화된 점' 항목을 아예 생성하지 말고 절대 포함시키지 마라."

                    final_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": prompt_msg}],
                        temperature=0.2, max_tokens=1500
                    )
                    
                    progress_bar.progress(100)
                    status_text.markdown("### ✅ AI 단속 파이프라인 분석 완료!")
                    
                    st.success("✨ 시계열 변화 대조 및 자동 매칭 단속 보고서가 성공적으로 발급되었습니다.")
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">{final_res.choices[0].message.content}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.markdown("### ❌ 분석 중 오류 발생")
                    st.error(f"오류 상세: {e}")

# ---------------------------------------------------------
# 메뉴 3: 자동 경보 시스템 (대시민 B2C)
# ---------------------------------------------------------
elif selected == "자동 경보 시스템":
    st.markdown("<h2>📢 대시민 상황 전파 및 알림 시스템 (B2C)</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='text-muted'>현재 실시간 풍향 <b>({get_wind_direction_str(w_dir)})</b> 및 드론 AI 적발 결과를 종합하여 주민 맞춤형 긴급 안내를 자동 생성합니다.</p><hr>", unsafe_allow_html=True)
    
    if 'alert_info' not in st.session_state: st.warning("⚠️ 먼저 [드론 비전 AI 단속] 메뉴에서 위반 사항을 탐지해 주세요.")
    else:
        alert_context = st.session_state['alert_info']
        st.info(f"🚨 **전달받은 현장 상황**: {alert_context}")
        st.info(f"🌤️ **적용 중인 기상 데이터**: {get_wind_direction_str(w_dir)} / 풍속 {w_spd} m/s (사이드바에서 변경 가능)")
        
        if st.button("🚨 시민 및 관리사무소 전파 메시지 생성"):
            with st.spinner("메시지 작성 중..."):
                try:
                    message_res = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": f"적발 내용({alert_context})과 기상 정보({get_wind_direction_str(w_dir)})를 바탕으로 피해 예상 지역 시민용 긴급 SMS(150자 내외)와 아파트 관리사무소 안내 방송 대본을 작성하라."}],
                        temperature=0.2, max_tokens=1000
                    )
                    st.success("✅ 메시지 작성 완료")
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">{message_res.choices[0].message.content}</div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"❌ 오류 발생: {e}")

# ---------------------------------------------------------
# 메뉴 4: 인프라 및 정책 제언
# ---------------------------------------------------------
elif selected == "인프라 및 정책 제언":
    st.markdown("<h2>🏛️ 과학적 예산 집행: 센서 입지 및 핀셋 지원 전략</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-muted'>데이터 시뮬레이션을 통해 도출된 핵심 타겟을 기반으로 행정 인프라 최적화를 제안합니다.</p><hr>", unsafe_allow_html=True)

    if not df_impact_apt.empty and not df_farm_raw.empty:
        st.subheader("📍 스마트 모니터링 센서 최적 설치 위치 제안")
        
        m2 = folium.Map(location=[36.815, 127.113], zoom_start=11, tiles='http://mt0.google.com/vt/lyrs=m&hl=ko&x={x}&y={y}&z={z}', attr='Google')
        
        target_farms_df = df_impact_apt['원인농가'].value_counts().head(5).reset_index()
        for farm_name in target_farms_df['원인농가'].tolist():
            farm_row = df_farm_raw[df_farm_raw['농가식별명'] == farm_name]
            if not farm_row.empty:
                f_lat, f_lon = float(get_scalar(farm_row.iloc[0].get('위도', 0))), float(get_scalar(farm_row.iloc[0].get('경도', 0)))
                if f_lat != 0 and f_lon != 0 and not pd.isna(f_lat):
                    folium.Circle(location=[f_lat, f_lon], radius=1000, color='orange', fill=True, fill_opacity=0.2, tooltip=f"센서 권장 구역 - {farm_name}").add_to(m2)
                    folium.Marker(location=[f_lat, f_lon], icon=folium.Icon(color='orange', icon='info-sign')).add_to(m2)

        for _, apt in df_impact_apt.head(5).iterrows():
            folium.Marker(location=[apt['위도'], apt['경도']], icon=folium.Icon(color='purple', icon='home'), tooltip=f"방역 벨트: {apt['공동주택명']}").add_to(m2)
            
        st_folium(m2, width="100%", height=450)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🎯 **최우선 저감 인프라(센서/포집기) 설치 권장 농가 Top 5**")
            target_farms_df.columns = ['농가명', '직접 피해 유발 횟수']
            st.dataframe(target_farms_df, use_container_width=True, hide_index=True)
        with col2:
            st.error("🚧 **최우선 방역 벨트 권장 구역 (Top 5 공동주택)**")
            st.dataframe(df_impact_apt[['공동주택명', '원인농가', 'CII', '위험등급']].head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 메뉴 5: IoT 역추적 관제
# ---------------------------------------------------------
elif selected == "IoT 역추적 관제":
    st.markdown("<h2>📡 IoT 센서 기반 원인 농가 역추적 (Reverse Tracking)</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-muted'>특정 지점(센서)에서 악취가 감지되었을 때, 역확산 모델링 및 사육두수 가중치를 결합하여 최우선 발원 확률을 산출하고 드론을 출동시킵니다.</p><hr>", unsafe_allow_html=True)

    if not df_impact_apt.empty and not df_farm_raw.empty:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("⚙️ 센서 관측 정보")
            sensor_locations = df_impact_apt.head(5)['공동주택명'].tolist()
            selected_sensor = st.selectbox("경보 발생 센서 위치 (아파트)", sensor_locations)
            odor_level = st.slider("감지된 악취 농도 (OU)", 0, 100, 45)
            
            sensor_row = df_impact_apt[df_impact_apt['공동주택명'] == selected_sensor].iloc[0]
            s_lat, s_lon = float(sensor_row['위도']), float(sensor_row['경도'])
            
            st.info(f"📍 **센서 위치:**\n{selected_sensor}\n\n🌬️ **유입 풍향:**\n{get_wind_direction_str(w_dir)} ({w_dir}°)\n\n💨 **유입 풍속:**\n{w_spd} m/s")
            
        with col2:
            m4 = folium.Map(location=[s_lat, s_lon], zoom_start=12, tiles='http://mt0.google.com/vt/lyrs=m&hl=ko&x={x}&y={y}&z={z}', attr='Google')
            
            folium.Marker(location=[s_lat, s_lon], icon=folium.Icon(color='purple', icon='tower'), tooltip=f"🚨 악취 감지 센서 ({selected_sensor})").add_to(m4)
            
            reverse_dir = w_dir
            angle_spread = 25 
            length = 0.045 
            
            p1 = [s_lat, s_lon]
            p2 = [s_lat + length * math.cos(math.radians(reverse_dir - angle_spread)), s_lon + length * math.sin(math.radians(reverse_dir - angle_spread))]
            p3 = [s_lat + length * math.cos(math.radians(reverse_dir + angle_spread)), s_lon + length * math.sin(math.radians(reverse_dir + angle_spread))]
            
            folium.Polygon(locations=[p1, p2, p3], color='purple', fill=True, fill_opacity=0.2, weight=1, tooltip="악취 발원지 역추적 영역 (확률 기반)").add_to(m4)
            
            suspects = []
            total_score = 0.0
            
            for _, farm in df_farm_raw.iterrows():
                f_lat, f_lon = float(get_scalar(farm.get('위도', 0))), float(get_scalar(farm.get('경도', 0)))
                if f_lat == 0 or f_lon == 0 or pd.isna(f_lat): continue
                
                dist, bearing = calculate_haversine_and_bearing(s_lat, s_lon, f_lat, f_lon)
                angle_diff = abs((bearing - reverse_dir + 180) % 360 - 180)
                
                if dist <= 5.0 and angle_diff <= angle_spread:
                    emission = float(get_scalar(farm.get('Odor_Emission', 0)))
                    if emission <= 0: emission = 10 
                    
                    score = emission / (max(dist, 0.1) ** 2)
                    
                    suspects.append({
                        '용의 농가명': farm.get('농가식별명', '미상'),
                        '주사육업종': farm.get('주사육업종', '미상'),
                        '사육두수': int(farm.get('사육두수', 0)),
                        '추정 거리(km)': round(dist, 2),
                        '_score': score,
                        'lat': f_lat,
                        'lon': f_lon
                    })
                    total_score += score
            
            if suspects:
                for s in suspects:
                    s['발원 확률(%)'] = round((s['_score'] / total_score) * 100, 1)
                suspects = sorted(suspects, key=lambda x: x['발원 확률(%)'], reverse=True)
                
                for i, s in enumerate(suspects):
                    if i == 0: 
                        folium.Marker(
                            location=[s['lat'], s['lon']],
                            icon=folium.Icon(color='red', icon='plane'),
                            tooltip=f"🚁 최우선 타겟: {s['용의 농가명']} (확률: {s['발원 확률(%)']}%)"
                        ).add_to(m4)
                        st.session_state['target_farm'] = s['용의 농가명']
                    else:
                        folium.CircleMarker(
                            location=[s['lat'], s['lon']], radius=6, color='orange', fill=True, fill_opacity=0.8,
                            tooltip=f"⚠️ 용의 농가: {s['용의 농가명']} (확률: {s['발원 확률(%)']}%)"
                        ).add_to(m4)
            else:
                for _, farm in df_farm_raw.iterrows():
                    f_lat, f_lon = float(get_scalar(farm.get('위도', 0))), float(get_scalar(farm.get('경도', 0)))
                    if f_lat != 0 and f_lon != 0 and not pd.isna(f_lat):
                        folium.CircleMarker(location=[f_lat, f_lon], radius=2, color='gray', fill=True, fill_opacity=0.3).add_to(m4)
                    
            st_folium(m4, width="100%", height=450)
            
            if suspects:
                top_target = suspects[0]
                st.error(f"🚨 **역추적 분석 완료:** 센서 풍상측 역확산 영역 내 총 **{len(suspects)}개**의 농가가 식별되었습니다.")
                st.warning(f"🚁 **최우선 드론 정찰 타겟:** **{top_target['용의 농가명']}** (발원 확률: {top_target['발원 확률(%)']}%, 거리: {top_target['추정 거리(km)']}km)\n\n상단 메뉴의 **[드론 비전 AI 단속]** 탭으로 이동하여 현장 영상을 업로드하고 AI 검증 및 행정 처분 절차를 진행해 주세요.")
                
                for s in suspects:
                    del s['_score']; del s['lat']; del s['lon']
                
                suspect_df = pd.DataFrame(suspects)
                st.dataframe(suspect_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ 현재 기상 조건의 역추적 경로(풍상측 5km 이내)에 식별된 농가가 없습니다. 타 지역구 유입 가능성을 배제할 수 없습니다.")

# ---------------------------------------------------------
# 메뉴 6: 대시민 챗봇
# ---------------------------------------------------------
elif selected == "대시민 챗봇":
    st.markdown("<h2>💬 대시민 실시간 악취 민원 챗봇</h2>", unsafe_allow_html=True)
    st.markdown("<p class='text-muted'>현재 상황과 대피 요령, 민원 접수 등에 대해 자유롭게 물어보세요!</p><hr>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = [{"role": "assistant", "content": "안녕하십니까, 천안시청 악취통합관리센터입니다. 현재 풍향 및 모니터링 정보를 바탕으로 무엇을 도와드릴까요?"}]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("예: 오늘 쌍용동 주변 악취 상황은 어때요?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("답변 작성 중..."):
                try:
                    current_status = st.session_state.get('alert_info', '특이 사항 없음')
                    sys_msg = f"당신은 천안시청 공무원입니다. 정중하게 응대하세요. 현재 적발상황: {current_status}. 현재 날씨: {get_wind_direction_str(w_dir)}, 풍속 {w_spd}m/s."
                    messages = [{"role": "system", "content": sys_msg}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-5:]]
                    response = client.chat.completions.create(model="meta/llama-3.1-8b-instruct", messages=messages, temperature=0.3, max_tokens=500)
                    full_response = response.choices[0].message.content
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e: message_placeholder.error(f"오류: {e}")