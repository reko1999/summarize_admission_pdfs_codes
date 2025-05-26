import os
import pdfplumber
import pytesseract
from PIL import Image
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from collections import defaultdict
from reportlab.lib.colors import black

# Tesseract 경로 설정 (Windows 예시, 환경에 맞게 수정)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# KoBART 모델 로드
model_name = "gogamza/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 1. 파일 처리 및 텍스트 추출 관련 메서드
def extract_text_from_pdf(pdf_path):
    text = ""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 먼저 텍스트 추출 시도
            page_text = page.extract_text()
            if page_text and not any(c in page_text for c in ['□', 'x']):  # 깨진 문자가 없으면 추가
                text += page_text + "\n"
            else:
                # 텍스트가 없거나 깨진 경우 OCR 사용
                try:
                    # 페이지를 이미지로 변환
                    page_image = page.to_image(resolution=300)  # DPI 300으로 고해상도
                    pil_image = page_image.original
                    # 한글 OCR 수행
                    ocr_text = pytesseract.image_to_string(pil_image, lang='kor', config='--psm 6')
                    if ocr_text.strip():
                        text += ocr_text + "\n"
                except Exception as e:
                    print(f"OCR failed for page: {e}")

            # 표 추출
            page_tables = page.extract_tables()
            if page_tables:
                for table in page_tables:
                    cleaned_table = []
                    for row in table:
                        cleaned_row = []
                        for cell in row:
                            if cell is None:
                                cell = ""
                            cell = str(cell).replace('\n', ' ').strip()
                            if len(cell) > 1000:
                                cell = cell[:1000] + "..."
                            cleaned_row.append(cell)
                        cleaned_table.append(cleaned_row)
                    tables.append(cleaned_table)

    # 텍스트가 여전히 비어 있거나 깨진 문자로 가득한 경우 경고
    if not text.strip() or any(c in text for c in ['□', 'x']):
        print(f"Warning: Extracted text from {pdf_path} may be incomplete or corrupted.")

    return text, tables


def preprocess_text(text):
    text = ' '.join(text.split())
    sentences = [s.strip() for s in text.split('. ') if s.strip() and not any(c in s for c in ['□', 'x'])]
    return '. '.join(sentences)


# 2. 텍스트 요약 관련 메서드
def summarize_text(text, target_ratio=0.1, chunk_size=512):
    if not text or len(tokenizer.encode(text, add_special_tokens=False)) == 0:
        return "No valid text for summarization."
    text = preprocess_text(text)
    sentences = text.split('. ')
    total_words = len(text.split())
    target_words = int(total_words * target_ratio)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length > chunk_size or not current_chunk:
            if current_chunk:
                chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    summaries = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        print(f"Chunk length: {len(tokenizer.encode(chunk, add_special_tokens=False))} tokens")
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=int(len(chunk.split()) * target_ratio * 2),
            min_length=10,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    final_summary = '. '.join(summaries)
    final_words = len(final_summary.split())
    if final_words > target_words or final_words < target_words // 2:
        inputs = tokenizer(final_summary, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=target_words,
            min_length=target_words // 4,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )
        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Final summary length: {len(final_summary.split())} words (Target: {target_words})")
    print("Final summary:", final_summary)
    return final_summary


# 3. 표 및 그래프 데이터 추출 관련 메서드
def extract_table_from_summary(summary):
    lines = summary.split('. ')
    table_data = []
    graph_data = {'labels': [], 'values': []}
    for line in lines:
        if any(keyword in line for keyword in ["정원", "학과", "일정"]) and any(char.isdigit() for char in line):
            parts = line.split()
            row = []
            label = None
            value = None
            for part in parts:
                if part.isdigit() or "명" in part or "일" in part:
                    value = part
                elif any(keyword in part for keyword in ["정원", "학과", "일정"]):
                    continue
                else:
                    label = part
                if label and value:
                    row = [label, value]
            if len(row) > 1:
                table_data.append(row)
                graph_data['labels'].append(row[0])
                numeric_value = ''.join(filter(str.isdigit, row[1]))
                graph_data['values'].append(int(numeric_value) if numeric_value else 0)
    return table_data if table_data else None, graph_data if graph_data['labels'] else None


# 4. PDF 생성 및 콘텐츠 추가 관련 메서드
def categorize_summary(summary):
    categories = defaultdict(list)
    lines = summary.split('. ')
    for line in lines:
        if "정원" in line:
            categories["모집 정원"].append(line)
        elif "일정" in line or "마감" in line:
            categories["입시 일정"].append(line)
        elif "전형" in line or "방법" in line:
            categories["전형 방법"].append(line)
        else:
            categories["기타"].append(line)
    return categories


def create_pdf(summary, tables, output_path):
    output_path_str = str(output_path)
    doc = SimpleDocTemplate(output_path_str, pagesize=A4)
    styles = getSampleStyleSheet()
    if 'Heading3' not in styles:
        styles.add(
            ParagraphStyle(name='Heading3', parent=styles['Heading2'], fontSize=12, spaceAfter=6, textColor=black))

    story = []
    story.append(Paragraph("입시요강 요약", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("핵심 내용", styles['Heading2']))
    categories = categorize_summary(summary)
    for category, lines in categories.items():
        if lines:
            story.append(Paragraph(category, styles['Heading3']))
            for line in lines:
                if line.strip():
                    story.append(Paragraph(f"• {line}", styles['Normal']))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))

    doc.build(story)


# 5. 파일 처리 및 워크플로우 관련 메서드
def process_files(input_folder):
    input_folder = Path(input_folder)
    output_folder = input_folder / "summaries"
    output_folder.mkdir(exist_ok=True)
    for file_path in input_folder.glob("*.pdf"):
        print(f"Processing {file_path.name}...")
        text, tables = extract_text_from_pdf(file_path)
        if not text.strip():
            print(f"No text extracted from {file_path.name}")
            continue
        summary = summarize_text(text)
        output_filename = f"(요약)_{file_path.name}"
        output_path = output_folder / output_filename
        create_pdf(summary, tables, output_path)


if __name__ == "__main__":
    input_folder = r"C:\입시-pdf-요약-테스트"
    process_files(input_folder)