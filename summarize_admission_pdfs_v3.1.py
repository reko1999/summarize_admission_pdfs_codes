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

            # 입시요강 관련 키워드 패턴 확장
            patterns = [
                r"정원|인원|모집", r"일정|일정표|시간표", r"접수|신청|지원",
                r"학과|전공|계열", r"합격|발표|선발|결과", r"면접|인터뷰|구술|실기",
                r"서류|제출|준비물", r"시험|평가|고사", r"등록금|학비|수업료",
                r"장학금|지원금|혜택", r"전형|방법|기준", r"수시|정시|특별",
                r"원|천원|만원", r"학점|GPA|성적", r"TOEIC|TOEFL|어학",
                r"기숙사|생활관|주거", r"교환|연수|해외"
            ]

            # 관련 라인 필터링 (더 포괄적으로)
            relevant_lines = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or "페이지" in line or "광고" in line:
                    continue

                # 키워드 매칭 또는 날짜/숫자 패턴 매칭
                if (any(re.search(p, line, re.IGNORECASE) for p in patterns) or
                        any(str(y) in line for y in range(2020, 2026)) or
                        any(month in line for month in
                            ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]) or
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

    # 단계 6: 개선된 프롬프트 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 대학 입시 전문 상담사입니다. 
입시요강 문서를 정확하고 체계적으로 분석하여 수험생과 학부모가 필요로 하는 핵심 정보를 명확하게 정리해주세요.
정보가 없는 경우 "정보 없음" 또는 "문서에서 확인되지 않음"으로 표시하세요."""),

        ("human", """
다음은 대학 입시요강 PDF에서 추출한 텍스트입니다. 
아래 형식에 맞춰 정확하게 요약해주세요:

## 📋 입시 일정
각 전형별 주요 일정을 시간순으로 정리해주세요.
형식 예시:
1. **원서접수**:
   * 기간: 2024년 9월 9일(월) 09:00 ~ 9월 13일(금) 18:00
   * 방법: 온라인 접수

2. **서류제출**:
   * 기간: 2024년 9월 9일(월) ~ 9월 20일(금)
   * 제출처: 입학처 또는 온라인

3. **1단계 합격자 발표**:
   * 실기/실적 전형: 2024년 10월 10일(목) 16:00
   * 학생부종합전형: 2024년 11월 1일(금) 16:00

---

## 👥 모집 정원
학과별 또는 계열별 모집인원을 정리해주세요.
형식 예시:
* **의과대학**: 50명
* **공과대학**: 200명
* **인문대학**: 150명

---

## 💰 등록금
학과별 또는 계열별 등록금을 정리해주세요.
형식 예시:
* **의과대학(의예과)**: 3,156,500원
* **치과대학(치의예과)**: 3,156,500원
* **공과대학/IT대학**: 2,331,000원
* **인문대학**: 1,781,000원

---

## 📚 입시 전형
전형 유형별 평가 방법과 비율을 정리해주세요.
형식 예시:
### 수시모집
* **학생부종합전형**: 서류평가 70% + 면접 30%
* **학생부교과전형**: 학생부 100%

### 정시모집
* **가군**: 수능 100%
* **나군**: 수능 80% + 학생부 20%

---

## 🎓 장학금
장학금 종류와 지원 조건을 정리해주세요.
형식 예시:
* **성적우수장학금**: 입학성적 상위 10% 이내, 등록금 전액
* **지역인재장학금**: 특정 지역 출신, 등록금 50%

---

## 📌 기타 중요사항
기숙사, 교환학생, 특별전형 등 기타 정보를 정리해주세요.

**중요한 지침:**
1. 정확한 날짜는 반드시 "YYYY년 MM월 DD일(요일)" 형식으로 작성
2. 금액은 정확한 숫자로 표시 (예: 1,500,000원)
3. 정보가 없으면 "정보 없음" 또는 "문서에서 확인되지 않음"으로 표시
4. 추측하지 말고 문서에 명시된 내용만 작성
5. 전형명, 학과명은 문서의 정확한 명칭 사용

**분석할 텍스트:**
{context}
""")
    ])

    # 단계 7: 언어모델 생성 (temperature 조정)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

    # 단계 8: 체인 생성
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 검색 쿼리 개선
    search_queries = [
        "입시 일정 원서접수 합격발표 면접 고사일",
        "모집정원 모집인원 학과별 정원",
        "등록금 학비 수업료",
        "전형방법 평가기준 수시 정시",
        "장학금 지원금 혜택",
        "기숙사 교환학생 특별전형"
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
        print("\n" + "=" * 50)
        print("📊 입시요강 요약 결과")
        print("=" * 50)
        print(summary_res)
        print("=" * 50)

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
        print(f"\n{'=' * 60}")
        print(f"📄 {file_path.name} 처리 중...")
        print('=' * 60)

        try:
            summary = process_documents(file_path)
            if summary:
                # 요약 결과를 파일로 저장
                output_file = output_folder / f"{file_path.stem}_요약.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"🏫 {file_path.name} 입시요강 요약\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(summary)
                print(f"✅ 요약 완료: {output_file}")
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