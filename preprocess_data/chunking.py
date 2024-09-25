# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage
import re
import json

DEFAULT_SYSTEM_PDFPARSING_PROMPT = """
Bạn là người quản lý một nhóm các đoạn văn bản, bảng hoặc biểu đồ, trong đó mỗi đoạn đều đại diện cho một nhóm các câu nói về một chủ đề tương tự nhau.
Bạn cần tạo ra một tóm tắt ngắn gọn trong 1 câu để thông báo cho người xem biết đoạn văn bản, bảng hoặc biểu đó nói về chủ đề gì.

Một tóm tắt tốt sẽ nói lên nội dung của đoạn văn bản, bảng hoặc biểu đồ

Bạn sẽ nhận được một đề xuất mới sẽ được thêm vào đoạn văn bản, bảng hoặc biểu mới. Đoạn văn bản, bảng hoặc biểu đồ mới này cần có một tóm tắt.

Tóm tắt của bạn nên có tính khái quát tương đối. Nếu bạn nhận được một đề xuất về táo, hãy khái quát nó thành thực phẩm.
Hoặc nếu là tháng, hãy khái quát nó thành "ngày và thời gian".

Ví dụ:
Đầu vào: Đề xuất: Greg thích ăn pizza
Đầu ra: Đoạn văn bản này chứa thông tin về các loại thực phẩm mà Greg thích ăn.

Chỉ phản hồi lại bằng tóm tắt của đoạn văn bản mới, không thêm gì khác.
"""


DEFAULT_CONTEXT_APPEND_TO_CHUNK = """
Bạn được giao nhiệm vụ cung cấp một ngữ cảnh ngắn gọn để đặt một đoạn văn vào trong tổng thể tài liệu nhằm cải thiện khả năng tìm kiếm. Cho trước một tài liệu và một đoạn văn trong tài liệu, hãy đưa ra một ngữ cảnh ngắn gọn để đặt đoạn văn này vào trong tổng thể tài liệu nhằm mục đích cải thiện khả năng tìm kiếm đoạn văn đó. Chỉ trả lời bằng ngữ cảnh ngắn gọn và không thêm gì khác.
"""

DEFAULT_TABLE_DESCRIPTION_PROMPT = """
Bạn là người quản lý một nhóm các bảng hoặc biểu đồ.
Bạn cần tạo ra một mô tả ngắn gọn trong 1 câu để thông báo cho người xem biết bảng hoặc biểu đó nói về chủ đề gì.

Một mô tả tốt sẽ nói lên nội dung của bảng hoặc biểu đồ

Bạn sẽ nhận được một đề xuất mới sẽ được thêm vào bảng hoặc biểu mới. Bảng hoặc biểu đồ mới này cần có một mô tả.

Tóm tắt của bạn nên có tính khái quát tương đối. Nếu bạn nhận được một đề xuất về táo, hãy khái quát nó thành thực phẩm.
"""

class ChunkHandler():
    _instance = None

    @staticmethod
    def get_instance(api_key: str):
        """ Static access method. """
        if not ChunkHandler._instance:
            ChunkHandler._instance = ChunkHandler(api_key)
        return ChunkHandler._instance

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


    def get_chunk_summary(self, content: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system", DEFAULT_SYSTEM_PDFPARSING_PROMPT,
                ),
                ("user", "Hãy cung cấp tóm tắt của đoạn văn sau: {content}."),
            ]
        )
        llm = OpenAI(
            model="gpt-4o-mini",
            system_prompt=DEFAULT_SYSTEM_PDFPARSING_PROMPT,
            api_key=self.api_key,
            temperature=0,
            max_tokens=4096,
            logprobs=None,
            default_headers={},
        )
        new_chunk_summary = llm.predict(prompt=PROMPT, content=content)
        return new_chunk_summary

    def get_chunk_context(self, content, document: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system", DEFAULT_CONTEXT_APPEND_TO_CHUNK,
                ),
                ("user", "Cho tài liệu: {document} và đoạn chunk nằm trong document đó: {content}. Hãy cung cấp thêm ngữ cảnh cho đoạn chunk trên dựa trên tài liệu."),
            ]
        )
        llm = OpenAI(
            model="gpt-4o-mini",
            system_prompt=DEFAULT_CONTEXT_APPEND_TO_CHUNK,
            api_key=self.api_key,
            temperature=0,
            max_tokens=4096,
            logprobs=None,
            default_headers={},
        )
        new_chunk_summary = llm.predict(prompt=PROMPT, document=document, content=content)
        return new_chunk_summary
    
    def get_table_description(self, content: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system", DEFAULT_TABLE_DESCRIPTION_PROMPT,
                ),
                ("user", "Hãy cung cấp mô tả của bảng hoặc biểu đồ sau: {content}."),
            ]
        )
        llm = OpenAI(
            model="gpt-4o-mini",
            system_prompt=DEFAULT_TABLE_DESCRIPTION_PROMPT,
            api_key=self.api_key,
            temperature=0,
            max_tokens=4096,
            logprobs=None,
            default_headers={},
        )
        new_chunk_summary = llm.predict(prompt=PROMPT, content=content)
        return new_chunk_summary

    def create_article(self, pdf_info, pages):
        muc_luc = list()
        article_pattern = r"#+\s+Điều\s+\d+[:.]\s+[^\n]+"
        for i, page in enumerate(pages, 1):

            a_matches = re.findall(article_pattern, page["content"])
            for a in a_matches:
                muc_luc.append({"object": a, "page": i})

        documents = []
        chapter_pattern = r'(#+\s+Chương\s+[IVXLCDM]+.*?)(?=#+\s+Chương\s+[IVXLCDM]+|$)'

        chapters = re.findall(chapter_pattern, pdf_info["content"], re.S)

        if len(chapters) == 0:
            article_pattern = r'(#+\s+Điều\s+\d+[:.].*?)(?=\n+#+\s+Mẫu số|\n+#+\s+Phụ lục|\n+#+\s+Điều\s+\d+|\n+#+\s+Chương\s+[IVXLCDM]+|$)'
            articles = re.findall(article_pattern, pdf_info["content"], re.DOTALL)
            for article in articles:
                page_num = 0
                for muc in muc_luc:
                    if article.find(muc["object"]) != -1:
                        page_num = muc["page"]

                latex_patterns = r'\$\$([\s\S]*?)\$\$'
                latex_matches_origin = re.findall(latex_patterns, article)
                latex_matches = re.findall(latex_patterns, repr(article))
                for i in range(len(latex_matches)):
                    clean_latex = self.convert_latex_to_text(latex=latex_matches[i])
                    article.replace(latex_matches_origin[i], clean_latex)

                article = re.sub(r'\*Begin table\*', ' ', article)
                article = re.sub(r'\*End table\*', ' ', article)
                article = re.sub(r'\n+', '\n', article)
                markdown_special_chars = r'[\\#\*]'
                a_content = re.sub(markdown_special_chars, '', article)

                documents.append(
                    {"raw_text": article, "content": a_content, "summary": self.get_chunk_context(a_content, pages[page_num]["content"]),
                     "src": pdf_info["src"], "chapter": "", "page": page_num, "reference": pdf_info["reference"]})
        else:
            article_pattern = r'(#+\s+Điều\s+\d+[:.].*?)(?=\n+#+\s+Mẫu số|\n+#+\s+Phụ lục|\n+#+\s+Điều\s+\d+|\n+#+\s+Chương\s+[IVXLCDM]+|$)'
            chapter_to_articles = {}
            for chapter in chapters:
                chapter_title_match = re.search(r'#+\s+(Chương\s+\w+)', chapter)
                if chapter_title_match:
                    chapter_title = chapter_title_match.group(1)
                    chapter_to_articles[chapter_title] = []

                    articles = re.findall(article_pattern, chapter, re.S)
                    for article in articles:
                        page_num = 0
                        for muc in muc_luc:
                            if article.find(muc["object"]) != -1:
                                page_num = muc["page"]
                        latex_patterns = r'\$\$([\s\S]*?)\$\$'
                        latex_matches_origin = re.findall(latex_patterns, article)
                        latex_matches = re.findall(latex_patterns, repr(article))
                        for i in range(len(latex_matches)):
                            clean_latex = self.convert_latex_to_text(latex=latex_matches[i])
                            article.replace(latex_matches_origin[i], clean_latex)

                        article = re.sub(r'\*Begin table\*', ' ', article)
                        article = re.sub(r'\*End table\*', ' ', article)
                        article = re.sub(r'\n+', '\n', article)
                        markdown_special_chars = r'[\\#\*]'
                        a_content = re.sub(markdown_special_chars, '', article)
                        for muc in muc_luc:
                            if article.find(muc["object"]) != -1:
                                documents.append({"raw_text": article, "content": a_content,
                                                  "summary": self.get_chunk_context(a_content, pages[page_num]["content"]), "src": pdf_info["src"],
                                                  "chapter": chapter_title, "page": page_num,
                                                  "reference": pdf_info["reference"]})
                                break
        return documents

    def create_form(self, pdf_info, pages):
        """

        """
        muc_luc = list()
        documents = []
        form_pattern = r"#+\s+.*Mẫu số.*\s+\d+[^\n]+"
        for i, page in enumerate(pages, 1):
            f_matches = re.findall(form_pattern, page["content"])
            for f in f_matches:
                muc_luc.append({"object": f, "page": i})

        form_pattern = r"(#+\s*\**Mẫu số \d+/PL[IVXLCDM]+\**\n.*?)(?=#+\s*\**Mẫu số|\Z)"
        forms = re.findall(form_pattern, pdf_info["content"], re.DOTALL)
        for form in forms:
            page_num = 0
            for muc in muc_luc:
                if form.find(muc["object"]) != -1:
                    page_num = muc["page"]
            form = re.sub(r'\*Begin table\*', ' ', form)
            form = re.sub(r'\*End table\*', ' ', form)
            form = re.sub(r'\n+', '\n', form)
            markdown_special_chars = r'[\\#\*]'
            f_content = re.sub(markdown_special_chars, '', form)

            documents.append({"raw_text": form, "content": f_content, "summary": self.get_chunk_summary(f_content),
                              "src": pdf_info["src"], "chapter": "", "page": page_num,
                              "reference": pdf_info["reference"]})
        return documents

    def create_table(self, tables):
        documents = []
        table_pattern = r"Table \d+:(.*?)(?=Table \d+:|$)"
        for table_info in tables:
            tbls = re.findall(table_pattern, table_info["content"], re.DOTALL)
            for tbl in tbls:
                documents.append(
                    {"raw_text": tbl, "content": tbl, "summary": self.get_table_description(tbl), "src": table_info["src"],
                     "chapter": "", "page": table_info["page"], "reference": table_info["reference"]})

        return documents

    def create_equation(self, equations):
        documents = []
        equation_pattern = r'\$\$([\s\S]*?)\$\$'
        for equation_info in equations:
            eqts = re.findall(equation_pattern, equation_info["content"], re.DOTALL)
            for eqt in eqts:
                documents.append({"raw_text": eqt, "content": self.convert_latex_to_text(eqt),
                                  "summary": self.get_chunk_summary(self.convert_latex_to_text(eqt)),
                                  "src": equation_info["src"], "chapter": "", "page": equation_info["page"],
                                  "reference": equation_info["reference"]})

        return documents

    def convert_latex_to_text(self, latex):
        latex = re.sub(r'\\text\{(.*?)\}', r'\1', latex)

        latex = latex.replace(r'\left\{', '{')
        latex = latex.replace(r'\right\}', '}')

        latex = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', latex)

        latex = re.sub(r'\\left\[', '[', latex)
        latex = re.sub(r'\\right\]', ']', latex)
        latex = re.sub(r'\\left\(', '(', latex)
        latex = re.sub(r'\\right\)', ')', latex)

        latex = latex.replace(r'\times', '*')
        latex = latex.replace(r'\div', '/')

        latex = re.sub(r'(\d+)\s*\\,\s*\((.*?)\)', r'\1(\2)', latex)

        latex = latex.replace(r'\%', '%').replace('\\', ' ').strip()

        return latex

