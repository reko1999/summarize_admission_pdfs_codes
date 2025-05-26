import os
import json
from pathlib import Path
import pdfplumber
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black

# OpenAI API 클라이언트 초기화 (API 키는 환경 변수 또는 직접 입력)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "입력하세요"))

# 한국어로 최적화된 요약 프롬프트
SUMMARY_PROMPT = """
다음은 한국어 입시요강 PDF에서 추출한 텍스트입니다. 아래 지침에 따라 한국어로 요약하세요:
1. 모집 정원: 학과별 정원을 키-값 쌍으로 정리. 정원이 명시되지 않은 학과는 "정원 미상"으로 표시.
2. 입시 일정: 주요 일정을 날짜순으로 리스트로 나열. 정보가 없으면 "정보 없음"으로 표시.
3. 전형 방법: 전형 유형과 평가 기준을 키-값 쌍으로 정리. 텍스트에 없는 정보는 포함하지 마세요.
4. 기타: 특별 전형 등 추가 정보를 리스트로 정리. 정보가 없으면 빈 리스트([])로 처리.
출력은 JSON 형식으로, 모든 키와 값을 한국어로 작성하세요. 텍스트에 명시되지 않은 정보는 추론하지 말고 "정원 미상", "정보 없음", 또는 빈 값으로 처리하세요.
예시 출력:
{{
  "모집 정원": {{"컴퓨터공학과": "50명", "경영학과": "정원 미상"}},
  "입시 일정": ["원서 접수: 2025.06.01~06.10", "정보 없음"],
  "전형 방법": {{"수시": "서류 70%, 면접 30%"}},
  "기타": []
}}
텍스트: {extracted_text}
"""

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트를 추출"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        if not text.strip():
            print(f"경고: {pdf_path}에서 텍스트를 추출하지 못했습니다.")
            return ""
        # print("text:", text)
        return text
    except Exception as e:
        print(f"{pdf_path} 텍스트 추출 오류: {e}")
        return ""

def summarize_with_llm(text):
    """LLM을 사용해 한국어로 텍스트 요약"""
    
    if not text.strip():
        return {"오류": "요약할 유효한 텍스트가 없습니다."}

    api_prompt = SUMMARY_PROMPT.format(extracted_text=text)
    print("api_prompt:", api_prompt)

    try:

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "당신은 한국어 입시요강을 요약하는 전문 도우미입니다. 모든 답변을 한국어로 제공하세요."},
                {"role": "user", "content": api_prompt}
            ],
            temperature=0.3,  # 일관된 출력
            max_tokens=1500   # 요약의 양 조절
        )
        summary = response.choices[0].message.content
        print("summary:", summary)
        # JSON 파싱
        try:
            return json.loads(summary)
        except json.JSONDecodeError:
            print("오류: LLM 출력이 유효한 JSON 형식이 아닙니다.")
            return {"오류": "LLM 출력이 JSON 형식이 아닙니다."}
    except Exception as e:
        print(f"LLM 요약 중 오류: {e}")
        return {"오류": str(e)}

def create_pdf(summary, output_path):
    """요약 데이터를 PDF로 생성"""
    output_path_str = str(output_path)
    doc = SimpleDocTemplate(output_path_str, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 사용자 정의 스타일 추가
    if 'Heading3' not in styles:
        styles.add(ParagraphStyle(
            name='Heading3',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            textColor=black
        ))
    
    story = []
    story.append(Paragraph("입시요강 요약", styles['Title']))
    story.append(Spacer(1, 12))

    # JSON 데이터 처리
    if "오류" in summary:
        story.append(Paragraph(f"요약 오류: {summary['오류']}", styles['Normal']))
    else:
        # 모집 정원
        if summary.get("모집 정원"):
            story.append(Paragraph("모집 정원", styles['Heading3']))
            data = [[k, v] for k, v in summary["모집 정원"].items()]
            # 표 생성
            table = Table(data, colWidths=[200, 100])
            table.setStyle([
                ('FONT', (0, 0), (-1, -1), 'Helvetica'),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT')
            ])
            story.append(table)
            story.append(Spacer(1, 6))
        
        # 입시 일정
        if summary.get("입시 일정"):
            story.append(Paragraph("입시 일정", styles['Heading3']))
            for item in summary["입시 일정"]:
                story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        # 전형 방법
        if summary.get("전형 방법"):
            story.append(Paragraph("전형 방법", styles['Heading3']))
            for k, v in summary["전형 방법"].items():
                story.append(Paragraph(f"• {k}: {v}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        # 기타
        if summary.get("기타"):
            story.append(Paragraph("기타", styles['Heading3']))
            for item in summary["기타"]:
                story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 6))
    
    try:
        doc.build(story)
        print(f"PDF 생성 완료: {output_path_str}")
    except Exception as e:
        print(f"PDF 생성 오류 {output_path_str}: {e}")

def process_files(input_folder):
    """입력 폴더의 모든 PDF 파일 처리"""
    input_folder = Path(input_folder)
    output_folder = input_folder / "summaries"
    output_folder.mkdir(exist_ok=True)
    
    for file_path in input_folder.glob("*.pdf"):
        print(f"{file_path.name} 처리 중...")
        text = extract_text_from_pdf(file_path)
        if not text:
            print(f"{file_path.name} 텍스트가 비어 있어 건너뜁니다.")
            continue
        summary = summarize_with_llm(text)
        output_filename = f"(요약)_{file_path.name}"
        output_path = output_folder / output_filename
        create_pdf(summary, output_path)

if __name__ == "__main__":
    input_folder = r"C:\입시-pdf-요약-테스트"
    process_files(input_folder)