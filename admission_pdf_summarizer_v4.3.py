from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pathlib import Path
import pdfplumber
import tiktoken
import re
from tiktoken import encoding_for_model
import os

# PDF 생성을 위한 라이브러리
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from datetime import datetime


def setup_korean_font():
    """한글 폰트 설정 - 시스템에 있는 한글 폰트를 찾아서 등록"""
    try:
        # Windows의 경우
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",  # 바탕
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('Korean', font_path))
                return 'Korean'
        
        # macOS의 경우
        mac_font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
        ]
        
        for font_path in mac_font_paths:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('Korean', font_path))
                return 'Korean'
                
        print("⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        return 'Helvetica'
        
    except Exception as e:
        print(f"⚠️  폰트 설정 중 오류: {e}. 기본 폰트를 사용합니다.")
        return 'Helvetica'

def create_pdf_summary(summary_text, output_path, university_name):
    """요약 텍스트를 PDF로 저장"""
    try:
        # 한글 폰트 설정
        korean_font = setup_korean_font()
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=inch * 0.7,
            leftMargin=inch * 0.7,
            topMargin=inch * 0.8,
            bottomMargin=inch * 0.8
        )
        
        # 스타일 정의
        styles = getSampleStyleSheet()
        
        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=korean_font,
            fontSize=18,
            spaceAfter=30,
            textColor=HexColor('#2c3e50'),
            alignment=1  # 중앙 정렬
        )
        
        # 부제목 스타일 (## 섹션)
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontName=korean_font,
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=HexColor('#34495e'),
            leftIndent=0
        )
        
        # 소제목 스타일 (### 섹션)
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading2'],
            fontName=korean_font,
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=HexColor('#7f8c8d'),
            leftIndent=10
        )
        
        # 본문 스타일
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontName=korean_font,
            fontSize=10,
            spaceAfter=6,
            leftIndent=0,
            rightIndent=0,
            leading=14
        )
        
        # 리스트 스타일
        list_style = ParagraphStyle(
            'CustomList',
            parent=styles['Normal'],
            fontName=korean_font,
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10,
            leading=13
        )
        
        # PDF 내용 구성
        story = []
        
        # 제목 추가
        story.append(Paragraph(f"🏫 {university_name} 입시요강 요약", title_style))
        story.append(Spacer(1, 12))
        
        # 생성 일시 추가
        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
        story.append(Paragraph(f"생성일시: {current_time}", body_style))
        story.append(Spacer(1, 20))
        
        # 요약 내용 파싱 및 추가
        lines = summary_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            # 마크다운 형식 처리
            if line.startswith('## '):
                # 주요 섹션 제목
                title = line[3:].strip()
                story.append(Paragraph(title, heading_style))
                
            elif line.startswith('### '):
                # 하위 섹션 제목
                title = line[4:].strip()
                story.append(Paragraph(title, subheading_style))
                
            elif line.startswith('* **') and '**:' in line:
                # 굵은 글씨가 있는 리스트 항목
                content = line[2:].strip()
                # **텍스트**: 부분을 처리
                if '**:' in content:
                    bold_part, rest = content.split('**:', 1)
                    bold_part = bold_part.replace('**', '')
                    formatted_content = f"<b>{bold_part}</b>: {rest.strip()}"
                else:
                    formatted_content = content
                story.append(Paragraph(f"• {formatted_content}", list_style))
                
            elif line.startswith('* ') or line.startswith('- '):
                # 일반 리스트 항목
                content = line[2:].strip()
                story.append(Paragraph(f"• {content}", list_style))
                
            elif line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
                # 번호 리스트 항목
                story.append(Paragraph(line, list_style))
                
            elif line.startswith('---'):
                # 구분선
                story.append(Spacer(1, 15))
                
            elif line and not line.startswith('#'):
                # 일반 본문
                # 굵은 글씨 처리
                formatted_line = line.replace('**', '<b>', 1).replace('**', '</b>', 1) if '**' in line else line
                story.append(Paragraph(formatted_line, body_style))
        
        # PDF 생성
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"❌ PDF 생성 중 오류: {e}")
        return False

def count_tokens(text):
    encoder = encoding_for_model("gpt-4o")
    return len(encoder.encode(text))

def extract_text_and_tables(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 텍스트 추출 (레이아웃 유지)
            page_text = page.extract_text(layout=True) or ""
            lines = page_text.splitlines()
            
            # 입시요강 관련 키워드 패턴을 핵심 항목에 맞춰 확장
            patterns = [
                # 입학자격 관련
                r"입학자격|지원자격|학력|졸업|재직|경력",
                # 모집정원 관련  
                r"정원|인원|모집|선발",
                # 입시일정 관련
                r"일정|일정표|시간표|접수|신청|지원|합격|발표|면접|고사|시험",
                # 전형방법 관련
                r"전형|방법|기준|평가|수시|정시|종합|교과",
                # 수능최저학력기준 관련
                r"최저|학력|기준|등급|수능|점수|성적",
                # 제출서류 관련
                r"서류|제출|구비|증명서|추천서|자기소개서",
                # 등록금 관련
                r"등록금|학비|수업료|납입|원|천원|만원",
                # 장학금 관련
                r"장학금|지원금|감면|혜택|성적우수|저소득",
                # 유의사항 관련
                r"유의|주의|중복|제한|금지|마감|환불",
                # 기타 중요사항 관련
                r"기숙사|교환|편입|복수|특별|해외|연수"
            ]
            
            # 관련 라인 필터링 (더 포괄적으로)
            relevant_lines = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or "페이지" in line or "광고" in line:
                    continue
                    
                # 키워드 매칭 또는 날짜/숫자 패턴 매칭
                if (any(re.search(p, line, re.IGNORECASE) for p in patterns) or
                    any(str(y) in line for y in range(2020, 2030)) or
                    any(month in line for month in ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]) or
                    any(day in line for day in [f"{i}일" for i in range(1, 32)]) or
                    re.search(r'\d+원|\d+명|\d+점', line) or  # 금액, 인원, 점수
                    re.search(r'\d{4}\.\d{1,2}\.\d{1,2}', line)):  # 날짜 형식
                    relevant_lines.append(line)
            
            text += "\n".join(relevant_lines) + "\n"
            
            # 표 추출
            tables = page.extract_tables()
            for table in tables:
                if table:  # 빈 테이블 제외
                    for row in table:
                        if row and any(cell for cell in row if cell):  # 빈 행 제외
                            text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
    
    return [Document(page_content=text)]

def process_documents(file_path):
    # 단계 1: 문서 로드
    docs = extract_text_and_tables(file_path)
    print(f"추출된 문서 수: {len(docs)}")
    
    # 추출된 텍스트 저장 (디버깅용)
    debug_file = Path(file_path).parent / f"{Path(file_path).stem}_extracted.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(docs[0].page_content)
    
    print(f"추출된 텍스트 미리보기:\n{docs[0].page_content[:1500]}...")
    print(f"추출된 텍스트 토큰 수: {count_tokens(docs[0].page_content)}")

    # 단계 2: 문서 분할 (청크 크기 조정)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,  # 청크 크기 줄임
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 문서 조각 수: {len(split_documents)}")

    # 단계 3: 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기 생성 (더 많은 조각 검색)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20, "fetch_k": 30}  # 더 많은 조각 검색
    )

    # 단계 6: 핵심 항목 중심 프롬프트 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 대학 입시 전문 상담사입니다. 
입시요강 문서를 정확하고 체계적으로 분석하여 수험생과 학부모가 필요로 하는 핵심 정보를 명확하게 정리해주세요.
정보가 없는 경우 "정보 없음" 또는 "문서에서 확인되지 않음"으로 표시하세요."""),
        
        ("human", """
다음은 대학 입시요강 PDF에서 추출한 텍스트입니다. 
아래 10개 핵심 항목에 맞춰 정확하게 요약해주세요:

## 🎓 입학 자격
지원 가능한 학력 조건과 자격 요건을 정리해주세요.
형식 예시:
* **기본 자격**: 고등학교 졸업 이상 또는 동등 학력 인정자
* **재직자 특별전형**: 재직 경력 3년 이상
* **외국인 특별전형**: 외국 국적자 또는 해외 고교 졸업자

---

## 👥 모집 정원
학과별 또는 전형별 모집인원을 정리해주세요.
형식 예시:
* **컴퓨터공학과**: 50명
* **경영학과**: 정원 미상
* **의예과**: 40명 (정시 20명, 수시 20명)

---

## 📋 입시 일정
주요 일정을 시간순으로 정리해주세요.
형식 예시:
1. **원서접수**:
   * 기간: 2024년 9월 9일(월) 09:00 ~ 9월 13일(금) 18:00
2. **서류제출**:
   * 기간: 2024년 9월 9일(월) 09:00 ~ 9월 20일(금) 18:00
3. **1단계 합격자 발표**:
   * 실기/실적 전형: 2024년 10월 10일(목) 16:00
4. **면접 및 고사일**:
   * 학생부종합전형: 2024년 11월 16일(토)
5. **최종 합격자 발표**:
   * 2024년 12월 20일(금) 15:00

---

## 📚 전형 방법
입시 전형 유형과 평가 기준을 정리해주세요.
형식 예시:
### 수시모집
* **학생부종합전형**: 서류평가 70% + 면접 30%
* **학생부교과전형**: 학생부 100%

### 정시모집
* **가군**: 수능 100%
* **나군**: 수능 80% + 학생부 20%

---

## 📊 대학수학능력시험 최저학력기준
수능 최저학력기준을 정리해주세요.
형식 예시:
* **인문계열**: 국어, 수학, 영어, 탐구 중 3개 영역 등급 합 7 이내
* **자연계열**: 국어, 수학, 영어, 과탐 중 3개 영역 등급 합 6 이내
* **의예과**: 국어, 수학, 영어, 과탐 모두 1등급

---

## 📄 제출 서류
지원 시 필요한 서류 목록을 정리해주세요.
형식 예시:
* **공통 서류**: 졸업증명서, 학교생활기록부
* **학생부종합전형**: 자기소개서, 추천서 1부
* **특별전형**: 재직증명서, 경력증명서
* **외국인 전형**: 외국인등록증, 한국어능력시험 성적표

---

## 💰 등록금 정보
학과별 또는 계열별 등록금을 정리해주세요.
형식 예시:
* **의과대학(의예과)**: 3,156,500원
* **치과대학(치의예과)**: 3,156,500원
* **공과대학/IT대학**: 2,331,000원
* **인문대학**: 1,781,000원

---

## 🎓 장학금 정보
장학금 종류와 지원 조건을 정리해주세요.
형식 예시:
* **성적우수장학금**: 입학성적 상위 10% 이내, 등록금 전액 지원
* **지역인재장학금**: 특정 지역 출신, 등록금 50% 감면
* **저소득층지원장학금**: 기초생활수급자, 등록금 전액 + 생활비 지원

---

## ⚠️ 유의사항
지원 시 주의할 사항을 정리해주세요.
형식 예시:
* **중복지원 제한**: 수시모집 6회 이내, 정시모집 3회 이내
* **서류 제출**: 마감일 18:00까지 도착분에 한함
* **면접 불참**: 면접 불참 시 불합격 처리
* **등록 포기**: 등록 포기 시 환불 규정 적용

---

## 📌 기타 중요사항
기숙사, 교환학생, 특별전형 등 기타 정보를 정리해주세요.
형식 예시:
* **기숙사**: 신입생 우선 배정, 월 30만원
* **교환학생**: 2학년부터 지원 가능, 연간 20명 선발
* **복수전공**: 2학년부터 신청 가능
* **편입학**: 3학년 편입 모집, 매년 3월

**중요한 지침:**
1. 정확한 날짜는 "YYYY년 MM월 DD일(요일) HH:MM" 형식으로 작성
2. 금액은 정확한 숫자와 단위로 표시 (예: 1,500,000원)
3. 정보가 없으면 "정보 없음" 또는 "문서에서 확인되지 않음"으로 표시
4. 추측하지 말고 문서에 명시된 내용만 작성
5. 전형명, 학과명은 문서의 정확한 명칭 사용
6. 각 항목별로 가능한 한 상세하게 작성

**분석할 텍스트:**
{context}
""")
    ])

    # 단계 7: 언어모델 생성 (temperature 조정)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

    # 단계 8: 체인 생성
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 검색 쿼리를 핵심 항목에 맞춰 개선
    search_queries = [
        "입학자격 지원자격 학력조건 졸업 재직자",
        "모집정원 모집인원 학과별 정원 선발인원",
        "입시일정 원서접수 합격발표 면접 고사일 시험일정",
        "전형방법 평가기준 수시 정시 전형유형",
        "수능최저학력기준 최저등급 수능등급 학력기준",
        "제출서류 구비서류 필요서류 증명서 추천서",
        "등록금 학비 수업료 납입금",
        "장학금 지원금 감면 혜택 성적우수",
        "유의사항 주의사항 중복지원 제한사항",
        "기숙사 교환학생 특별전형 편입학 복수전공"
    ]
    
    # 모든 검색 결과 통합
    all_docs = []
    for query in search_queries:
        docs = retriever.invoke(query)
        all_docs.extend(docs)
    
    # 중복 제거 (내용 기준)
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    
    context = format_docs(unique_docs)
    
    print(f"검색된 문서 조각 수: {len(unique_docs)}")
    print(f"컨텍스트 토큰 수: {count_tokens(context)}")

    # 체인 실행
    chain = prompt | llm | StrOutputParser()
    
    try:
        summary_res = chain.invoke({"context": context})
        print("\n" + "="*50)
        print("📊 입시요강 요약 결과")
        print("="*50)
        print(summary_res)
        print("="*50)
        
        return summary_res
    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return None

def process_files(input_folder):
    input_folder = Path(input_folder)
    output_folder = input_folder / "summaries"
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print("PDF 파일을 찾을 수 없습니다.")
        return

    for file_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"📄 {file_path.name} 처리 중...")
        print('='*60)
        
        try:
            summary = process_documents(file_path)
            if summary:
                # 대학 이름 추출 (파일명에서)
                university_name = file_path.stem
                
                # TXT 파일로도 저장 (기존 기능 유지)
                txt_output_file = output_folder / f"{file_path.stem}_요약.txt"
                with open(txt_output_file, "w", encoding="utf-8") as f:
                    f.write(f"🏫 {file_path.name} 입시요강 요약\n")
                    f.write("="*60 + "\n\n")
                    f.write(summary)
                print(f"✅ TXT 요약 완료: {txt_output_file}")
                
                # PDF 파일로 저장 (새로운 기능)
                pdf_output_file = output_folder / f"{file_path.stem}_요약.pdf"
                if create_pdf_summary(summary, pdf_output_file, university_name):
                    print(f"✅ PDF 요약 완료: {pdf_output_file}")
                else:
                    print(f"❌ PDF 생성 실패: {pdf_output_file}")
                    
            else:
                print(f"❌ {file_path.name} 요약 실패")
        except Exception as e:
            print(f"❌ {file_path.name} 처리 중 오류: {e}")

if __name__ == "__main__":
    # 환경변수에서 OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경변수를 설정해주세요.")
        exit(1)
    
    input_folder = r"C:\입시-pdf-요약-테스트"
    
    if not Path(input_folder).exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        exit(1)
    
    process_files(input_folder)
    print("\n🎉 모든 작업 완료!")
