import fitz
from PIL import Image
import io
import base64
from io import BytesIO
import requests
import time
import re


class ParseHandler():
    _instance = None
    prompt_system = """You are an AI assistant that processes images containing legal content. Your task is to transcribe the content from the image into markdown format with the following rules:

                        1. Use `#` for titles if you see the name of a Law or Decree.
                        2. Use `##` if you see the word "Chương" or "Mẫu số" at the beginning of a sentence.
                        3. Use `###` if you see the word "Mục" at the beginning of a sentence.
                        4. Use `####` if you see the word "Điều" at the beginning of a sentence.
                        5. Do not use any '#' for a), b), c),... etc or 1), 2), 3),...etc
                        6. Only respond with the markdown content, without any additional explanation or description.
                        7. Don't annotate ```markdown\n  in the response

                                    """
    detect_equation_prompt = """You are an expert in equation dectection. If the input image have any equations, response "Equation". If there aren't any equations in the image, respponse "No equation"."""
    detect_table_prompt = """You are an expert in table dectection. If the input image have any tables, response "Table". If there aren't any tables in the image, respponse "No table"."""
    parse_table_prompt = """You are an expert in table dectection. Focus on each table in the input image and response the markdown of each table with the index of each table in the begin of table (i.e. Table 1:, Table 2:, ...etc) without any additional explanation or description. Try to keep all the text of each table at the right cell in the image and do not annotate ```markdown."""
    parse_equation_prompt = """You are an expert in equation dectection. Focus on each equation in the input image:
                                - The equation can be on multiple lines, detect all the lines of the equation. 
                                - Convert the equation in the image into a single latex block with all expressions inside a pair of dollar signs ($$). Try to get both legs before and after the equal sign(=) (i.e. $$ \text{Gross bonus amount} = \text{Bonus} \times \left(100 - \frac{\text{Date of absent in a year}}{22 \times 12}\right) $$)
                                - Response the equation with the index of each equation in the begin of equation (i.e. Equation 1:, Equation 2:, ...etc) without any additional explanation or description.
     """

    @staticmethod
    def get_instance(api_key):
        """ Static access method. """
        if not ParseHandler._instance:
            ParseHandler._instance = ParseHandler(api_key)
        return ParseHandler._instance

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def pdf_to_images(self, file_stream, pdf_path, zoom_x=2.0, zoom_y=2.0):
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        image_base64s = []

        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_base64s.append(base64.b64encode(image_bytes).decode("utf-8"))
        return image_base64s

    def parse_table(self, image_base64):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.parse_table_prompt
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Parse table in this image"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }
        cost = 0
        tables = ""
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            tables = response.json()['choices'][0]['message']['content']
            cost = response.json()['usage']["prompt_tokens"] * 0.000005 + response.json()['usage'][
                "completion_tokens"] * 0.000015
        except:
            print("Lỗi trong việc phân tích JSON:")
        return tables, cost

    def parse_equation(self, image_base64):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.parse_equation_prompt
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Parse equation in this image"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }
        cost = 0
        equations = ""
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            equations = response.json()['choices'][0]['message']['content']
            cost = response.json()['usage']["prompt_tokens"] * 0.000005 + response.json()['usage'][
                "completion_tokens"] * 0.000015
        except:
            print("Lỗi trong việc phân tích JSON:")
        return equations, cost

    def detect_table(self, image_base64):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.detect_table_prompt
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Detect table in this image"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }
        response = ""
        detect_cost = 0
        table_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            detect_cost = table_response.json()['usage']["prompt_tokens"] * 0.000005 + table_response.json()['usage'][
                "completion_tokens"] * 0.000015
            if table_response.json()['choices'][0]['message']['content'] == "Table":
                response, cost = self.parse_table(image_base64=image_base64)
                detect_cost += cost
        except:
            print("Lỗi trong việc phân tích JSON:")

        return response, detect_cost

    def detect_equation(self, image_base64):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.detect_equation_prompt
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Detect equation in this image"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }
        response = ''
        detect_cost = 0
        equation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            detect_cost = equation_response.json()['usage']["prompt_tokens"] * 0.00000015 + \
                          equation_response.json()['usage']["completion_tokens"] * 0.0000006
            if equation_response.json()['choices'][0]['message']['content'] == "Equation":
                response, cost = self.parse_equation(image_base64=image_base64)
                detect_cost += cost
        except:
            print("Lỗi trong việc phân tích JSON:", equation_response.text)

        return response, detect_cost

    def concatenate_pages(self, pages, file_name):
        pdf_content = ""
        for page in pages:
            i = page["page"]
            page_content = page["content"] + f"\n\n"
            pdf_content += page_content
        pdf_info = {"content": pdf_content, "reference": pages[0]["reference"], "src": file_name}
        return pdf_info

    def parse_pdf(self, image_base64s, file_name):
        pages = []
        tables = []
        equations = []
        law_name = ""
        for i in range(0, len(image_base64s)):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt_system
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe the content from this image into markdown format"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64s[i]}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4096
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            print(response)
            try:
                cost = response.json()['usage']["prompt_tokens"] * 0.00000015 + response.json()['usage'][
                    "completion_tokens"] * 0.0000006
                page_content = response.json()['choices'][0]['message']['content']
                if i == 0:
                    pattern = [r'(BỘ LUẬT\s*\n)(.*)', r'(LUẬT\s*\n)(.*)', r'(NGHỊ ĐỊNH\s*\n)(.*)',
                               r'(QUY ĐỊNH\s*.*\n)(.*)']
                    text = ""
                    for p in pattern:
                        text = re.findall(p, page_content)
                        if len(text) > 0:
                            law_name = text[0][0].strip() + " " + text[0][1].strip()
                            break
                detected_tables, tables_cost = self.detect_table(image_base64=image_base64s[i])
                detected_equations, equations_cost = self.detect_equation(image_base64=image_base64s[i])
                cost += tables_cost + equations_cost
                pages.append({"page": i + 1, "content": page_content, "reference": law_name, "cost": cost})
                if detected_tables != "":
                    tables.append({"page": i + 1, "content": detected_tables, "reference": law_name, "src": file_name})

                if detected_equations != "":
                    equations.append(
                        {"page": i + 1, "content": detected_equations, "reference": law_name, "src": file_name})
                time.sleep(2)
            except:
                print("Lỗi trong việc phân tích JSON:")

        pdf_info = self.concatenate_pages(pages=pages, file_name=file_name)
        return pages, pdf_info, tables, equations
