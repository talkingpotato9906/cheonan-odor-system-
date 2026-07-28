import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def generate_alert_and_report(vision_result_json, weather_info, target_apartment):
    """
    비전 AI 결과 + 기상 데이터 + 대상 아파트 정보를 조합하여
    LLM(Llama 3.1 70B 등)으로 행정 보고서와 주민 경보 텍스트를 자동 생성합니다.
    """
    
    # RAG를 위한 환경부 악취방지법 간이 가이드라인 데이터베이스 (Context 주입)
    regulatory_context = """
    [대한민국 악취방지법 가이드라인]
    - 제8조 (배출허용기준): 악취배출시설에서 배출되는 악취의 희석배수는 공업지역 1000 이하, 기타지역 500 이하(부지경계선 기준)여야 한다.
    - 제14조 (개선명령): 기준 초과 시 지자체장은 기간을 정하여 개선을 명할 수 있다.
    - 주민 피해 최소화를 위해 야간 정체 기온역전 현상 시 인근 공동주택 단지에 사전 알림 및 환기시설 폐쇄 조치 권고를 시행한다.
    """

    prompt = f"""
    당신은 천안시 광역 악취통합관리센터의 대응 지원 AI입니다.
    아래 [데이터 입력] 정보를 바탕으로 [출력 양식]에 맞춰 실시간 보고서 및 알림을 작성하세요.

    [데이터 입력]
    1. 비전 AI 분석 결과: {json.dumps(vision_result_json, ensure_ascii=False)}
    2. 현재 기상 상황: {json.dumps(weather_info, ensure_ascii=False)}
    3. 타격 예상 아파트: {json.dumps(target_apartment, ensure_ascii=False)}
    4. 관련 법령 참고: {regulatory_context}

    [출력 양식]
    ---
    ■ 1. 천안시청 환경과 행정조치 보고서 (공무원용)
    - 상황 요약:
    - 법적 근거 및 단속 가이드라인: (참고 법령을 인용하여 구체적으로 작성)
    - 긴급 조치 권고사항:

    ■ 2. 공동주택 실시간 스마트 조기경보 (주민 알림 및 관리실 안내 방송용)
    - 안내 방송 대본: (친절하고 신속한 대피 요령을 포함한 3줄 이내 방송 대본)
    - 주민 푸시 알림 문자:
    ---
    """

    print("📢 기상 및 분석 데이터를 매핑하여 실시간 대응 시나리오를 생성하는 중...")
    
    try:
        response = client.chat.completions.create(
            # 텍스트 생성과 추론에 뛰어난 Llama 3.1 70b Instruct 사용
            model="meta/llama-3.1-70b-instruct", 
            messages=[
                {"role": "system", "content": "당신은 행정 문서와 재난 알림 작성에 특화된 환경부 소속 전문 보좌관입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# --- 가상의 파이프라인 연동 데모 테스트 ---
if __name__ == "__main__":
    # 1. 이전 단계에서 sample_farm2를 통해 얻은 비전 분석 결과
    mock_vision_result = {
        "detected_objects": ["폐수 수조(Lagoon)", "개방형 정화조"],
        "risk_level": "8/10",
        "reasoning": "덮개가 없는 대규모 적자색 분뇨 저장조가 노출되어 있으며, 대기 중으로 악취 물질이 무방비하게 확산될 가능성이 매우 높음."
    }
    
    # 2. 앞서 구축한 마스터 기상 데이터에서 긁어온 실시간 기상 상태 (예시)
    mock_weather = {
        "관측소": "아산(AWS)",
        "풍향": "서남서풍(WSW) -> 천안 동부 방면으로 기류 이동",
        "풍속": "1.8 m/s (대기 정체 및 악취 이동 최적 속도)",
        "시간": "22:00 (야간 기온역전 현상 발생)"
    }
    
    # 3. 앞서 거리 분석 및 타격 횟수 분석으로 도출해 낸 위험군 아파트 (예시)
    mock_apartment = {
        "공동주택명": "천안 목천연합초원아파트",
        "거리": "2.4 km (풍하측 방향 15도 이내 직접 노출)"
    }

    report = generate_alert_and_report(mock_vision_result, mock_weather, mock_apartment)
    print("\n" + "="*60)
    print("🔥 [최종 파이프라인 데모] 시스템 자동 생성 결과 🔥")
    print("="*60)
    print(report)