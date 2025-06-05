# 입시 요강 PDF 요약 시스템 - LangChain 활용
from pathlib import Path

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader

# from langchain_summarize_test_2 import save_extracted_docs

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEBUG_TEXT = ""

class AdmissionSummarizer:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name = "gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vectorstore = None

        # 1. 섹션별 검색 쿼리 정의
        self.section_queries = {
            "입학자격": [
                "입학 지원 자격과 학력 조건은?",
                "재직자 특별전형 지원 요건은?",
                "외국인 특별전형 자격 기준은?",
                "성인학습자 입학 자격은?",
                "편입학 지원 자격 조건은?"
            ],
            "모집정원": [
                "학과별 모집정원은 몇 명인가?",
                "수시 정시 전형별 모집인원은?",
                "정원내 정원외 모집 구분과 인원은?",
                "특별전형별 선발 인원은?",
                "총 모집정원과 학부별 배정 인원은?"
            ],
            "입시일정": [
                "원서접수 기간은 언제인가?",
                "면접과 실기고사 일정은?",
                "합격발표일은 언제인가?",
                "등록 및 추가합격 일정은?",
                "전형별 주요 일정 차이는?",
            ],
            "전형방법": [
                "수시전형 평가방법과 반영비율은?",
                "정시전형 평가기준과 비율은?",
                "면접고사 평가 방법과 비중은?",
                "실기고사 평가 기준은?",
                "학생부 반영방법과 비율은?"
            ],
            "수능최저": [
                "수능 최저학력기준은?",
                "인문계열 수능 최저등급 기준은?",
                "자연계열 수능 최저등급 기준은?",
                "예체능계열 수능 최저기준은?",
                "수능 반영영역과 선택과목은?"
            ],
            "제출서류": [
                "공통 제출서류 목록은?",
                "수시전형 제출서류는?",
                "정시전형 제출서류는?",
                "특별전형 추가 제출서류는?",
                "서류 제출방법과 유의사항은?"
            ],
            "등록금": [
                "인문사회계열 등록금은?",
                "자연공학계열 등록금은?",
                "예체능계열 등록금은?",
                "입학금과 수업료는 각각 얼마인가?",
                "기타 납부금과 총 등록비용은?"
            ],
            "장학금": [
                "신입생 장학금 종류와 조건은?",
                "성적우수 장학금 기준과 지원규모는?",
                "소득연계 장학금 조건은?",
                "특별전형 장학금 혜택은?",
                "장학금 중복수혜와 유지조건은?"
            ],
            "유의사항": [
                "중복지원 금지 및 제한사항은?",
                "원서접수 및 서류제출 주의사항은?",
                "전형료 납부와 환불 규정은?",
                "합격자 등록 관련 유의사항은?",
                "기타 입시 관련 주의사항은?"
            ],
            "기타사항": [
                "기숙사 신청방법과 비용은?",
                "교환학생 프로그램 정보는?",
                "편입학 관련 추가 정보는?",
                "신입생 오리엔테이션과 지원제도는?",
                "학사 운영과 캠퍼스 시설 정보는?"
            ]
        }

        # 2. 섹션별 상세 프롬프트 템플릿
        self.section_prompts = {
            "입학자격": """
                다음 입시 요강 정보를 바탕으로 입학자격 정보를 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 일반전형, 특별전형별 학력 조건
                - 재직자/외국인 등 특별 요건
                - 편입학 자격 조건
                - 지원 제한 사항

                **관련 정보:**
                {context}

                **입학자격 요약:**
                """,

            "모집정원": """
                다음 입시 요강 정보를 바탕으로 모집정원 정보를 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 계열별 모집인원
                - 전형별 세부 인원
                - 정원 내/외 구분
                - 학과별 정확한 모집인원

                **관련 정보:**
                {context}

                **모집정원 요약:**
                """,

            "입시일정": """
                다음 입시 요강 정보를 바탕으로 입시일정을 시간순으로 정확하게 요약해주세요.

                **요약 기준:**
                - 원서접수 ~ 합격발표까지 시간순 정리
                - 각 전형별 일정 구분
                - 중요 마감일과 주의사항
                - 면접/실기 등 개별 일정

                **관련 정보:**
                {context}

                **입시일정 요약:**
                """,

            "전형방법": """
                다음 입시 요강 정보를 바탕으로 전형방법을 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 수시/정시별 평가 방법
                - 학생부/수능/면접 등 반영 비율
                - 서류평가 기준
                - 실기/면접 평가 방법

                **관련 정보:**
                {context}

                **전형방법 요약:**
                """,

            "수능최저": """
                다음 입시 요강 정보를 바탕으로 수능 최저학력기준을 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 학과별/모집단위별 등급 요건
                - 반영 영역과 조합
                - 특별전형별 기준
                - 한국사/탐구 등 세부 조건

                **관련 정보:**
                {context}

                **수능 최저학력기준 요약:**
                """,

            "제출서류": """
                다음 입시 요강 정보를 바탕으로 제출서류를 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 전형별 필수 제출서류
                - 선택 제출서류
                - 서류 제출 방법과 기한
                - 추가 서류 요구사항

                **관련 정보:**
                {context}

                **제출서류 요약:**
                """,

            "등록금": """
                다음 입시 요강 정보를 바탕으로 등록금 정보를 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 학과/계열별 정확한 등록금 금액
                - 입학금과 수업료 구분
                - 추가 납부 비용
                - 등록금 납부 방법과 기한

                **관련 정보:**
                {context}

                **등록금 정보 요약:**
                """,

            "장학금": """
                다음 입시 요강 정보를 바탕으로 장학금 정보를 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 장학금 종류별 조건과 금액
                - 성적/소득 기준
                - 신청 방법과 기한
                - 지원 규모와 혜택

                **관련 정보:**
                {context}

                **장학금 정보 요약:**
                """,

            "유의사항": """
                다음 입시 요강 정보를 바탕으로 유의사항을 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 중복지원 제한 사항
                - 마감일과 시간 관련 주의사항
                - 전형별 특별 유의사항
                - 기타 중요한 제한사항

                **관련 정보:**
                {context}

                **유의사항 요약:**
                """,

            "기타사항": """
                다음 입시 요강 정보를 바탕으로 기타 중요사항을 정확하고 체계적으로 요약해주세요.

                **요약 기준:**
                - 기숙사 정보와 신청 방법
                - 교환학생/해외연수 프로그램
                - 편입학 관련 정보
                - 기타 학교 혜택과 프로그램

                **관련 정보:**
                {context}

                **기타 중요사항 요약:**
                """
        }

        # 3. 최종 통합 프롬프트 템플릿
        self.final_prompt = PromptTemplate(
            input_variables=["sections"],
            template="""
            다음은 입시 요강에서 추출한 섹션별 정보입니다. 
            이를 바탕으로 완전하고 체계적인 입시 요강 요약서를 작성해주세요.

            {sections}
            
            **최종 입시 요강 요약서:**

            # 🎓 입학 자격
            {입학자격 내용을 여기에 정리}

            # 👥 모집 정원
            {모집정원 내용을 여기에 정리}

            # 📋 입시 일정
            {입시일정 내용을 여기에 정리}

            # 📚 전형 방법
            {전형방법 내용을 여기에 정리}

            # 📊 수능 최저학력기준
            {수능최저 내용을 여기에 정리}

            # 📄 제출 서류
            {제출서류 내용을 여기에 정리}

            # 💰 등록금 정보
            {등록금 내용을 여기에 정리}

            # 🎓 장학금 정보
            {장학금 내용을 여기에 정리}

            # ⚠️ 유의사항
            {유의사항 내용을 여기에 정리}

            # 📌 기타 중요사항
            {기타사항 내용을 여기에 정리}
            """
        )

    def load_and_process_pdf(self, pdf_path):
        """PDF 로드 및 벡터스토어 생성"""
        # PDF 로드
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()

        # 텍스트 분할 (입시 요강에 맞게 조정)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        print(f"분할된 문서 조각 수: {len(docs)}")

        # chunk_index = 0
        # for chunk in docs:
        #     chunk_index += 1
        #     print(f"chunk.page_content[{chunk_index}] 길이: {len(chunk.page_content)}")
        #     print(f"chunk.page_content[{chunk_index}]:")
        #     print(chunk.page_content)

        self.save_extracted_docs(pdf_path, docs)

        # 벡터스토어 생성
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def save_extracted_docs(self, pdf_path, docs):
        debug_file = Path(pdf_path).parent / f"{Path(pdf_path).stem}_extracted_docs.txt"
        global DEBUG_TEXT
        with open(debug_file, "w", encoding="utf-8") as f:
            for doc in docs:
                DEBUG_TEXT+= doc.page_content + '\n'
                f.write(doc.page_content + '\n')

    def retrieve_section_info(self, section_name):
        """섹션별 정보 검색 및 요약"""
        if not self.vectorstore:
            raise ValueError("PDF를 먼저 로드해주세요.")

        section_info = []
        queries = self.section_queries.get(section_name, [])

        # 여러 쿼리로 검색하여 정보 수집
        for i, query in enumerate(queries):
            print(f"\n{'=' * 50}")
            print(f"Query {i + 1}: {query}")
            print(f"{'=' * 50}")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.75,
                    "k": 7
                }
            )
            docs = retriever.invoke(query)
            # docs = self.vectorstore.similarity_search_with_score(
            #     query,
            #     k=10  # 더 많이 가져와서 threshold 적용 후 상위 5개 선택
            # )
            print(f"검색된 문서 개수: {len(docs)}")

            # 각 문서의 유사도 점수와 내용 확인
            # for j, doc in enumerate(docs):
            #     print(f"\n--- 문서 {j + 1} ---")
            #
            #     # 메타데이터에서 유사도 점수 확인 (vectorstore에 따라 다를 수 있음)
            #     if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            #         print(f"유사도 점수: {doc.metadata['score']:.4f}")
            #
            #     # 문서 내용 미리보기 (처음 200자)
            #     # content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            #     content_preview = doc.page_content
            #     print(f"문서 내용 미리보기: {content_preview}")

            section_info.extend([doc.page_content for doc in docs])
            print(f"\n현재까지 수집된 총 섹션 수: {len(section_info)}")

        # 중복 제거
        unique_info = list(set(section_info))
        context = "\n\n".join(unique_info)
        print("context:", context)

        # 섹션별 요약 생성
        prompt = self.section_prompts.get(section_name, "")
        if prompt:
            formatted_prompt = prompt.format(context=context)
            response = self.llm(formatted_prompt)

            # content 속성이 있는지 확인 후 추출
            if hasattr(response, 'content'):
                summary = response.content
            else:
                summary = str(response)  # 혹시 모를 경우 대비

            return summary

        return context

        # return 1

    def generate_full_summary(self, pdf_path):
        """전체 입시 요강 요약 생성"""
        # PDF 처리
        self.load_and_process_pdf(pdf_path)

        # 섹션별 요약 생성
        sections = {}
        for section_name in self.section_queries.keys():
            print(f"Processing {section_name}...")
            sections[section_name] = self.retrieve_section_info(section_name)
        # for section_name in self.section_queries.keys():
        #     print("section name:", section_name)
        #     print(f"sections[{section_name}]: {sections[section_name]}")

        return sections

        # # 최종 요약서 생성
        # sections_text = "\n\n".join([
        #     f"**{name}:**\n{content}"
        #     for name, content in sections.items()
        # ])
        #
        # print("sections_text:")
        # print(sections_text)
        #
        # return sections_text

        # final_summary = self.final_prompt.format(sections=sections_text)
        # return self.llm(final_summary)

        # return 1


class SimpleSummaryEvaluator:

    def evaluate(self, original_text, summary_text):
        """
        4가지 핵심 지표로 요약 품질 평가
        """
        results = {
            'semantic_similarity': self.semantic_similarity(original_text, summary_text),
            'info_preservation': self.info_preservation(original_text, summary_text),
            'readability': self.readability(summary_text),
            'compression_efficiency': self.compression_efficiency(original_text, summary_text)
        }

        # 종합 점수 (정보 보존율에 높은 가중치)
        weights = [0.25, 0.40, 0.20, 0.15]  # 의미유사도, 정보보존, 가독성, 압축효율
        results['overall_score'] = sum(score * weight for score, weight in zip(results.values(), weights))

        return results

    def semantic_similarity(self, original, summary):
        """의미적 유사도 (TF-IDF 코사인 유사도)"""
        try:
            texts = [original, summary]
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity * 100
        except:
            return 0

    def info_preservation(self, original, summary):
        """핵심 정보 보존율 (TF-IDF 기반)"""
        try:
            # 원본에서 상위 중요 단어 추출
            vectorizer = TfidfVectorizer(max_features=30, stop_words=None)
            original_vector = vectorizer.fit_transform([original])

            # 중요 단어들
            feature_names = vectorizer.get_feature_names_out()
            important_words = set(feature_names)

            # 요약에서 단어 추출
            summary_words = set(summary.split())

            # 보존된 중요 단어 비율
            preserved_words = important_words & summary_words
            preservation_ratio = len(preserved_words) / len(important_words) if important_words else 0

            return preservation_ratio * 100

        except:
            return 0

    def readability(self, text):
        """가독성 점수"""
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        if not sentences:
            return 0

        # 평균 문장 길이 (한국어 적정: 15-25자)
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        length_score = max(0, 100 - abs(avg_length - 20) * 3)

        # 복잡도 (접속사 사용률)
        conjunctions = ['그리고', '하지만', '그러나', '또한', '따라서']
        complex_ratio = sum(1 for s in sentences if any(c in s for c in conjunctions)) / len(sentences)
        complexity_score = max(0, 100 - abs(complex_ratio - 0.3) * 200)

        return (length_score + complexity_score) / 2

    def compression_efficiency(self, original, summary):
        """압축 효율성 (적정 압축률: 20-40%)"""
        ratio = len(summary) / len(original) if len(original) > 0 else 0

        if 0.2 <= ratio <= 0.4:
            return 100
        elif ratio < 0.2:
            return 60  # 과도한 압축
        else:
            return max(0, 100 - (ratio - 0.4) * 150)  # 압축 부족


def test_evaluator(original, summary):
    evaluator = SimpleSummaryEvaluator()

    results = evaluator.evaluate(original, summary)

    print("=== 요약 품질 평가 ===")
    print(f"의미적 유사도: {results['semantic_similarity']:.1f}점")
    print(f"정보 보존율: {results['info_preservation']:.1f}점")
    print(f"가독성: {results['readability']:.1f}점")
    print(f"압축 효율성: {results['compression_efficiency']:.1f}점")
    print(f"종합 점수: {results['overall_score']:.1f}점")
    print()


import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


class SimpleAdmissionPDFGenerator:
    def __init__(self, font_path=None):
        """
        입시요강 PDF 생성기 초기화 (표 생성 없음)

        Args:
            font_path (str): 한국어 폰트 파일 경로 (옵션)
        """
        self.font_path = font_path
        self.setup_fonts()

        # 색상 정의
        self.colors = {
            'primary': HexColor('#2E5984'),  # 네이비 블루
            'secondary': HexColor('#4A90E2'),  # 라이트 블루
            'accent': HexColor('#F8F9FA'),  # 연한 회색
            'text': HexColor('#2C3E50'),  # 다크 그레이
            'header': HexColor('#1A365D'),  # 헤더 색상
        }

        # 스타일 설정
        self.setup_styles()

    def setup_fonts(self):
        """폰트 설정"""
        try:
            if self.font_path and os.path.exists(self.font_path):
                pdfmetrics.registerFont(TTFont('Korean', self.font_path))
                self.font_name = 'Korean'
            else:
                # 기본 한국어 폰트 경로들 시도
                font_paths = [
                    '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS
                    'C:/Windows/Fonts/malgun.ttf',  # Windows
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Linux
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        pdfmetrics.registerFont(TTFont('Korean', path))
                        self.font_name = 'Korean'
                        break
                else:
                    self.font_name = 'Helvetica'
                    print("한국어 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
        except Exception as e:
            self.font_name = 'Helvetica'
            print(f"폰트 설정 중 오류 발생: {e}")

    def setup_styles(self):
        """문서 스타일 설정"""
        self.styles = getSampleStyleSheet()

        # 제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontName=self.font_name,
            fontSize=24,
            textColor=self.colors['header'],
            spaceAfter=30,
            alignment=TA_CENTER,
            borderWidth=2,
            borderColor=self.colors['primary'],
            borderPadding=10,
            backColor=self.colors['accent']
        ))

        # 섹션 헤더 스타일
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontName=self.font_name,
            fontSize=16,
            textColor=white,
            backColor=self.colors['primary'],
            borderPadding=8,
            spaceAfter=12,
            spaceBefore=20,
            alignment=TA_LEFT
        ))

        # 본문 스타일
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=11,
            textColor=self.colors['text'],
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            firstLineIndent=0,
            leading=16
        ))

        # 리스트 스타일
        self.styles.add(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10,
            textColor=self.colors['text'],
            spaceAfter=6,
            leftIndent=20,
            bulletIndent=10,
            leading=14
        ))

        # 강조 텍스트 스타일 (콜론이 있는 정보성 텍스트용)
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10,
            textColor=self.colors['text'],
            spaceAfter=6,
            leftIndent=15,
            backColor=self.colors['accent'],
            borderPadding=5,
            leading=14
        ))

    def format_section_content(self, content):
        """섹션 내용을 포맷팅 (표 생성 없음)"""
        elements = []

        # 내용을 문자열로 변환
        content_str = str(content)
        lines = content_str.split('\n')
        current_paragraph = []

        for line in lines:
            line = line.strip()
            if not line:
                # 빈 줄이면 현재 문단 완료
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.styles['CustomBody']))
                    current_paragraph = []
                continue

            # 리스트 아이템 감지
            if (line.startswith('• ') or
                    line.startswith('- ') or
                    re.match(r'^\d+\.\s', line)):

                # 이전 문단 완료
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.styles['CustomBody']))
                    current_paragraph = []

                # 리스트 마커 제거하고 정리
                clean_line = re.sub(r'^[•\-]\s*|^\d+\.\s*', '', line).strip()
                if clean_line:
                    elements.append(Paragraph(f"• {clean_line}", self.styles['ListItem']))

            # 콜론이 있는 정보성 라인 (강조 처리)
            elif ':' in line and not line.startswith('**'):
                # 이전 문단 완료
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.styles['CustomBody']))
                    current_paragraph = []

                elements.append(Paragraph(line, self.styles['InfoText']))

            # 일반 텍스트
            else:
                current_paragraph.append(line)

        # 마지막 문단 처리
        if current_paragraph:
            elements.append(Paragraph(' '.join(current_paragraph), self.styles['CustomBody']))

        return elements

    def generate_pdf_from_sections(self, sections_dict, output_path, title="경북대학교 입시요강 요약"):
        """
        섹션 딕셔너리에서 직접 PDF 파일 생성 (표 생성 없음)

        Args:
            sections_dict (dict): 섹션명을 키로, 내용을 값으로 하는 딕셔너리
            output_path (str): 출력 PDF 파일 경로
            title (str): PDF 제목
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=25 * mm,
            bottomMargin=25 * mm
        )

        # 스토리 요소들
        story = []

        # 제목
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))

        # 섹션별 처리
        for section_title, section_content in sections_dict.items():
            if not section_content or not str(section_content).strip():
                continue

            # 섹션 헤더
            story.append(Paragraph(section_title, self.styles['SectionHeader']))

            # 섹션 내용 포맷팅 및 추가
            elements = self.format_section_content(section_content)
            story.extend(elements)

            # 섹션 간 여백
            story.append(Spacer(1, 15))

        # PDF 생성
        try:
            doc.build(story)
            print(f"PDF가 성공적으로 생성되었습니다: {output_path}")
            return True
        except Exception as e:
            print(f"PDF 생성 중 오류 발생: {e}")
            return False

def extract_title_from_pdf_path(pdf_path):
    # 파일명만 추출 (확장자 제거)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # '(홈)_' 제거
    title = filename.replace('(홈)_', ' ')

    # 연도 패턴 찾기 (4자리 숫자)
    year_match = re.search(r'(\d{4})', title)
    if year_match:
        year = year_match.group(1)
        # 연도를 '연도학년도'로 변경
        title = title.replace(year, f'{year}학년도')

    # 맨 뒤에 '요약' 추가
    title = title + ' 요약'

    return title

# def process_pdfs(input_folder):
#     output_folder = input_folder / "summaries"
#     output_folder.mkdir(exist_ok=True)
#
#     pdfs = list(input_folder.glob('*.pdf'))
#     if not pdfs:
#         print("PDF 파일을 찾을 수 없습니다.")
#         return
#
#     for pdf_path in pdfs:
#         print(f"\n{'=' * 60}")
#         print(f"📄 {pdf_path.name} 처리 중...")
#         print('=' * 60)
#
#         try:
#             summary =

# 사용 예시
# if __name__ == "__main__":
#
#     pdf_path = "C:\입시-pdf-요약-테스트\경북대(홈)_2025 수시 모집요강.pdf"
#     new_txt_path = pdf_path.replace(".pdf", "_(요약).txt")
#     new_pdf_path = pdf_path.replace(".pdf", "_(요약).pdf")
#     title = extract_title_from_pdf_path(pdf_path)
#
#     # 초기화
#     summarizer = AdmissionSummarizer(openai_api_key=your_api_key)
#
#     # PDF 요약 생성
#     summary = summarizer.generate_full_summary(pdf_path)
#
#     # 최종 요약서 생성
#     summary_text = "\n\n".join([
#         f"**{name}:**\n{content}"
#         for name, content in summary.items()
#     ])
#
#     print("summary_text:")
#     print(summary_text)
#
#     # print("summary:")
#     # print(summary)
#
#     print("DEBUG TEXT:", DEBUG_TEXT)
#     test_evaluator(original=DEBUG_TEXT, summary=summary_text)
#
#     # 결과 저장
#     with open(new_txt_path, "w", encoding="utf-8") as f:
#         f.write(summary_text)
#
#     # PDF 생성기 인스턴스 생성
#     generator = SimpleAdmissionPDFGenerator()
#
#     # PDF 생성
#     generator.generate_pdf_from_sections(
#         sections_dict=summary,
#         output_path=new_pdf_path,
#         title=title
#     )
#
#     print("입시 요강 요약이 완료되었습니다!")

import os
from pathlib import Path
import traceback


def process_pdf_folder(input_folder, your_api_key):
    """폴더 내 모든 PDF 파일을 처리하는 함수"""
    input_folder = Path(input_folder)

    if not input_folder.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        return

    # PDF 파일 찾기
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ {input_folder}에서 PDF 파일을 찾을 수 없습니다.")
        return

    print(f"📁 발견된 PDF 파일: {len(pdf_files)}개")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    # 출력 폴더 설정
    output_folder = input_folder / "summaries"
    output_folder.mkdir(exist_ok=True)
    print(f"📂 요약 파일들은 '{output_folder}' 폴더에 저장됩니다.")

    # 처리 결과 추적
    success_count = 0
    failure_count = 0
    results = []

    # 각 PDF 파일 처리
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'=' * 60}")
        print(f"📄 [{i}/{len(pdf_files)}] {pdf_path.name} 처리 중...")
        print('=' * 60)

        try:
            # 출력 파일 경로 설정
            new_txt_path = output_folder / f"{pdf_path.stem}_(요약).txt"
            new_pdf_path = output_folder / f"{pdf_path.stem}_(요약).pdf"

            title = extract_title_from_pdf_path(str(pdf_path))

            # 초기화
            summarizer = AdmissionSummarizer(openai_api_key=your_api_key)

            # PDF 요약 생성
            print(f"📄 {pdf_path.name} 요약 생성 중...")
            summary = summarizer.generate_full_summary(str(pdf_path))

            # 최종 요약서 생성
            summary_text = "\n\n".join([
                f"**{name}:**\n{content}"
                for name, content in summary.items()
            ])

            print(f"📝 요약 텍스트 길이: {len(summary_text)} 문자")

            # 디버그 및 평가 (필요시)
            if 'DEBUG_TEXT' in globals():
                print("DEBUG TEXT:", DEBUG_TEXT)
                test_evaluator(original=DEBUG_TEXT, summary=summary_text)

            # TXT 결과 저장
            with open(new_txt_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"✅ TXT 저장 완료: {new_txt_path}")

            # PDF 생성기 인스턴스 생성
            generator = SimpleAdmissionPDFGenerator()

            # PDF 생성
            generator.generate_pdf_from_sections(
                sections_dict=summary,
                output_path=str(new_pdf_path),
                title=title
            )
            print(f"✅ PDF 저장 완료: {new_pdf_path}")

            success_count += 1
            results.append((pdf_path.name, True, f"{pdf_path.name} 처리 완료"))

        except Exception as e:
            error_msg = f"{pdf_path.name} 처리 중 오류: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"상세 오류:\n{traceback.format_exc()}")
            failure_count += 1
            results.append((pdf_path.name, False, error_msg))

    # 최종 결과 출력
    print(f"\n{'=' * 60}")
    print("🎉 배치 처리 완료!")
    print('=' * 60)
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {failure_count}개")
    print(f"📊 총 처리: {len(pdf_files)}개")

    if failure_count > 0:
        print(f"\n❌ 실패한 파일들:")
        for filename, success, message in results:
            if not success:
                print(f"  - {filename}: {message}")


if __name__ == "__main__":
    # API 키 설정
    your_api_key = "your_api_key"  # 실제 API 키로 변경하세요

    # 처리할 폴더 경로
    input_folder = r"C:\입시-pdf-요약-테스트"

    # 폴더 내 모든 PDF 파일 배치 처리
    process_pdf_folder(input_folder, your_api_key)

    print("\n🎉 모든 작업 완료!")
