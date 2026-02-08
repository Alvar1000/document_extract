import paddle
from paddleocr import PaddleOCR
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class DocumentProcessor:
    """Класс для инициализации и хранения модели для повторного вызова"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("Загрузка моделей...")

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ru"
        )

        # Загружаем LLM один раз
        MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self._initialized = True
        print("Модели загружены!")

    def clean_fio(self, text: str) -> str:
        text = re.sub(r"[^А-ЯЁа-яё\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_fio_from_text(self, text: str) -> str:
        """ФИО из OCR-текста"""
        system_prompt = (
            "Ты — эксперт по анализу документов. "
            "Текст получен через OCR (распознавание), он содержит мусор, ошибки и написан КАПСОМ. "
            "Твоя задача: найти и исправить ФИО человека. "
            "Игнорируй одиночные буквы-мусор перед фамилией (например 'Е ИВАНОВ' -> 'ИВАНОВ'). "
            "Верни ТОЛЬКО ФИО в формате: Фамилия Имя Отчество. Больше ничего не пиши."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Текст для анализа:\n{text}"}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return result

    def process_document(self, image_path: str) -> str:
        """OCR + извлечение ФИО"""
        result = self.ocr.predict(input=image_path)

        texts = []
        for res in result:
            texts.extend(res['rec_texts'])

        full_text = " ".join(texts)

        # удаляем лишние символы(Оставляю чисто заглавные, так как в доках фио с большой только)
        dirty_text = self.clean_fio(full_text)
        fio = self.extract_fio_from_text(dirty_text)

        return fio

processor = DocumentProcessor()

def extract_fio(image_path: str) -> str:
    return processor.process_document(image_path)