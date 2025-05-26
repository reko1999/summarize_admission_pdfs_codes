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
            # 텍스트 추출
            page_text = page.extract_text(layout=True) or ""
            lines = page_text.splitlines()
            # patterns: 입시요강 관련 키워드를 유연하게 매칭하기 위한 정규 표현식 패턴 리스트
            # - 예: "정원|인원"은 "모집 정원", "선발 인원" 등을 포함
            # - "접수|신청|지원"은 "원서 접수", "지원 접수" 등을 포괄
            patterns = [
                r"정원|인원", r"일정|일정표", r"접수|신청|지원", r"학과",
                r"합격|발표|선발", r"면접|인터뷰|구술", r"서류|제출", r"시험|평가"
            ]
            # relevant_lines: 불필요한 라인("페이지", "광고" 포함)을 제외하고,
            # 키워드(patterns) 또는 날짜 패턴(2020~2025, 1월~12월, 1일~31일)이 포함된 라인만 선택
            # - "페이지 1" → 제외
            # - "원서 접수: 2025.06.01" → 포함 (키워드 "접수" 및 날짜 "2025" 매칭)
            # - "6월 1일 면접" → 포함 (월/일 패턴 매칭)
            relevant_lines = [line for line in lines if not ("페이지" in line or "광고" in line) and
                             (any(re.search(p, line) for p in patterns) or
                              any(line.strip().startswith(str(y)) for y in range(2020, 2026)) or
                              any(month in line for month in ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]) or
                              any(day in line for day in [f"{i}일" for i in range(1, 32)]))]
            text += "\n".join(relevant_lines) + "\n"
            # pdfplumber로 표 추출 (입시 일정표 등)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    text += " | ".join([str(cell) for cell in row if cell]) + "\n"
    return [Document(page_content=text)]

def process_documents(file_path):
    # 단계 1: 문서 로드
    docs = extract_text_and_tables(file_path)
    print("추출된 문서 수:", len(docs))
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(docs[0].page_content)
    print("추출된 텍스트 미리보기:", docs[0].page_content[:2000])
    print("추출된 텍스트 토큰 수:", count_tokens(docs[0].page_content))

    # 단계 2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기 생성
    # retriever: FAISS 벡터스토어에서 쿼리에 따라 관련 문서 조각(k=10)을 검색
    # - search_type="mmr": 중복을 줄이고 다양한 조각 반환
    # - k=10: 최대 10개 조각 반환 (입시 일정, 정원 등 포함 가능성 높임)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15})
    # context: retriever가 쿼리로 검색한 문서 조각을 문자열로 결합
    # - 쿼리: "입시 일정 | 모집 일정 | ..."로 입시 일정 관련 정보 검색
    # - 각 조각(page_content)을 "\n"으로 연결
    # - 프롬프트의 {context}에 삽입되어 gpt-4o가 요약할 데이터 제공
    # - 예: "컴퓨터공학과: 50명\n원서 접수: 2025.06.01"
    context = "\n".join([d.page_content for d in retriever.invoke("입시 일정 | 모집 일정 | 원서 접수 | 합격 발표 | 일정표 | 신청 | 지원 | 발표 | 인터뷰")])
    print("Retriever 결과:")
    for d in retriever.invoke("입시 일정 | 모집 일정 | 원서 접수 | 합격 발표 | 일정표 | 신청 | 지원 | 발표 | 인터뷰"):
        print("-----------------------------")
        print(d.page_content[:500])
        print("페이지 컨텐츠 길이:", len(d.page_content))
    print("컨텍스트 토큰 수:", count_tokens(context))

    # 단계 6: 프롬프트 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "입시 상담 전문가로, 텍스트에서 핵심 정보를 정확히 요약하세요. 정보 없으면 '텍스트에서 관련 정보 미발견' 표시."),
        ("human", """
        다음은 입시요강 PDF 텍스트입니다. 아래 지침에 따라 요약하세요:

        1. **학과별 모집 정원**: 학과별(예: 컴퓨터공학과) 정원을 목록화. 전형별 정원 제외, 미상은 "정원 미상".
        2. **입시 일정**: 원서 접수, 면접 등 날짜순 목록. "YYYY.MM.DD" 형식, 정보 없으면 "텍스트에서 관련 정보 미발견".
        3. **전형 방법**: 전형 유형(수시, 정시)과 평가 기준(서류, 면접 비율).
        4. **기타**: 특별 전형, 장학금 등. 정보 없으면 "텍스트에서 관련 정보 미발견".

        **출력 지침**:
        - 간결한 한국어, JSON 제외.
        - 항목은 ---로 구분.
        - 추론 금지, "정원 미상" 또는 "텍스트에서 관련 정보 미발견".
        - 날짜는 "YYYY.MM.DD".

        **예시 출력**:
        학과별 모집 정원:
        - 컴퓨터공학과: 50명
        - 경영학과: 정원 미상
        ---
        입시 일정:
        - 원서 접수: 2025.06.01~2025.06.10
        - 면접: 2025.06.20
        ---
        전형 방법:
        - 수시: 서류 70%, 면접 30%
        - 정시: 수능 100%
        ---
        기타:
        - 특별 전형: 지역인재 우선선발
        - 장학금: 성적 우수자 전액 지원

        **텍스트**:
        {context}
        """)
    ])
    print("프롬프트 토큰 수:", count_tokens(prompt.format(context=context, question="")))

    # 단계 7: 언어모델 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 단계 8: 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 실행
    # question = "입시요강을 위 프롬프트 지침에 따라 요약해줘"
    question = "합격자 발표날 언제인지 알려줘"
    summary_res = chain.invoke(question)
    print("요약 결과:")
    print(summary_res)

    return summary_res

def process_files(input_folder):
    input_folder = Path(input_folder)
    output_folder = input_folder / "summaries"
    output_folder.mkdir(exist_ok=True)

    for file_path in input_folder.glob("*.pdf"):
        print(f"{file_path.name} 처리 중...")
        summary = process_documents(file_path)
        if not summary:
            print(f"{file_path.name} 요약 결과가 비어 있어 건너뜁니다.")
            continue

if __name__ == "__main__":
    input_folder = r"C:\입시-pdf-요약-테스트"
    process_files(input_folder)
    print("모든 작업 완료")
