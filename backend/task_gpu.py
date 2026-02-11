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
        print("OCR загрузилась")

        # Загружаем LLM один раз
        print("Загрузка ллм")
        MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Токенизатор загружен")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=torch.float16
        )

        self._initialized = True
        print("Модели загружены!")

    def extract_fio_from_text(self, text: str) -> str:
        """Извлекает ФИО из грязного OCR-текста."""
        system_prompt = (
            "Ты — модуль нормализации данных СТС. Твоя цель — извлечь ФИО собственника на русском языке.\n\n"
            "ИНСТРУКЦИЯ:"
            "1. Ищи текст после ключевых слов: '(владелец)'."
            "2. В документах СТС часто идет дублирование: Русское Слово Английский Транслит. "
            "   Твоя задача — ПРОПУСТИТЬ английские слова (транслит) и оставить только русские."
            "3. Исправляй 'грязный' OCR: Латинские буквы (H, B, C, P, M, K, E, T, Y, X, A, O), попавшие внутрь русских слов, замени на кириллицу. "
            "   (Пример: 'BАСИЛЬЕВHА' -> 'ВАСИЛЬЕВНА')."
            "4. Верни строго 3 слова: Фамилия Имя Отчество."
            "5. Возвращай только то, что видишь в тексте, не додумывай окончания, фио в тексте написаны полностью"
            "6. Обязательно проверь, не изменил ли ты буквы в ФИО после извлечении их из текста, бери фамилию из текста, не придумывай"
            "7. Проверяй себя, выделенная тобой фио точно есть в тексте или ты сгенерировал лишние токены, лишние не генерируй"
            "\n"
            "ПРИМЕР РАБОТЫ (обучающий):"
            "Вход: 77 RUS ... СОБСТВЕННИК (владелец) КYЗНЕЦ KUZNETC MAPИЯ MARIA ИBAHOBHA MOCKBA УЛ..."
            "Выход: КУЗНЕЦ МАРИЯ ИВАНОВНА"
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
                max_new_tokens=35,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        clean_result = result.replace("Выход:", "").replace("Ответ:", "").strip()

        return clean_result

    def process_document(self, image_path: str) -> str:
        """OCR + извлечение ФИО"""
        result = self.ocr.predict(input=image_path)

        texts = []
        for res in result:
            texts.extend(res['rec_texts'])

        full_text = " ".join(texts)

        # Обрезаем текст до 140 символов
        full_text = full_text[:140]

        fio = self.extract_fio_from_text(full_text)

        return fio


processor = DocumentProcessor()


def extract_fio(image_path: str) -> str:
    return processor.process_document(image_path)