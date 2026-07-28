import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APITimeoutError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.2-90b-vision-instruct"
FALLBACK_MODEL = "meta/llama-3.2-11b-vision-instruct"
REQUEST_TIMEOUT = 300.0
MAX_RETRIES = 3
RETRY_DELAY = 10

IMAGE_PATHS = [
    Path(__file__).parent / "sample_farm.jpeg",
    Path(__file__).parent / "sample_farm2.jpeg",
]

SYSTEM_PROMPT = (
    "이미지 내 악취 유발 가능 시설(폐수, 개방형 정화조 등)을 분석해 "
    "JSON 형식(detected_objects, risk_level, reasoning)으로 반환해 줘"
)

RETRYABLE_ERRORS = (InternalServerError, APITimeoutError, RateLimitError)


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mime_types.get(image_path.suffix.lower(), "image/jpeg")


def analyze_image(client: OpenAI, image_path: Path, model: str = MODEL) -> str:
    base64_image = encode_image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "첨부된 이미지를 분석해 주세요.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except RETRYABLE_ERRORS as error:
            last_error = error
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(
                    f"[{image_path.name}] {type(error).__name__} 발생 "
                    f"({attempt}/{MAX_RETRIES}). {wait}초 후 재시도..."
                )
                time.sleep(wait)
            else:
                break

    raise RuntimeError(
        f"[{image_path.name}] {model} 분석 실패: {last_error}\n"
        "504/타임아웃은 NVIDIA NIM 서버 과부하 또는 90B 모델 응답 지연으로 "
        "자주 발생합니다. 잠시 후 다시 시도하거나, "
        f"경량 모델({FALLBACK_MODEL})로 자동 재시도합니다."
    ) from last_error


def analyze_image_with_fallback(client: OpenAI, image_path: Path) -> tuple[str, str]:
    try:
        return analyze_image(client, image_path, MODEL), MODEL
    except RuntimeError:
        print(f"[{image_path.name}] {FALLBACK_MODEL}로 대체 분석을 시도합니다...")
        return analyze_image(client, image_path, FALLBACK_MODEL), FALLBACK_MODEL


def print_result(image_path: Path, result: str, model: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"분석 대상: {image_path.name} (모델: {model})")
    print("=" * 60)
    print(result)

    try:
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        print("\n--- Parsed JSON ---")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        pass


def main() -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

    missing = [path for path in IMAGE_PATHS if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {missing_names}")

    client = OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    for image_path in IMAGE_PATHS:
        result, model = analyze_image_with_fallback(client, image_path)
        print_result(image_path, result, model)


if __name__ == "__main__":
    main()
