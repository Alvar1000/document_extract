## Извлечение ФИО из фото документа (СТС)

Простой сервис, который по загруженному фото документа (например, СТС) пытается извлечь ФИО владельца.  
Проект состоит из backend‑API на FastAPI и frontend‑интерфейса на Streamlit.

Продакшн‑развёртывание доступно по адресу:  
http://45.159.211.61:8501/

---

## Архитектура

- Frontend: Streamlit
  - Позволяет загрузить фото (jpg, jpeg, png).
  - Показывает превью загруженного документа.
  - По кнопке отправляет файл на backend и отображает извлечённое ФИО.

- Backend: FastAPI
  - Эндпоинты:
    - POST /process-document — принимает файл, запускает обработку и возвращает ФИО.
    - GET /health — простой health‑check для Docker.
  - Принимает файл, сохраняет во временный файл, запускает пайплайн распознавания и удаляет временный файл.

- Обработка документа
  - OCR: PaddleOCR (настройка для русского языка).
  - LLM: модель Qwen/Qwen2.5-3B-Instruct через transformers и torch.
  - Логика:
    1. OCR извлекает весь текст из изображения.
    2. Текст очищается (оставляются только кириллические символы и пробелы).
    3. В LLM отправляется промпт, который просит найти и аккуратно восстановить ФИО в формате
       «Фамилия Имя Отчество».
    4. Модель возвращает только ФИО, которое отдаётся клиенту.

---

## Стек технологий

- Backend
  - Python 3.11
  - FastAPI
  - Uvicorn
  - PaddlePaddle
  - PaddleOCR
  - PyTorch (torch)
  - Transformers
  - Pillow

- Frontend
  - Streamlit
  - Requests
  - Pillow

- Инфраструктура
  - Docker
  - Docker Compose

---

## Структура проекта

document_extract/
├── docker-compose.yaml
├── README.txt
│
├── backend/
│   ├── Dockerfile
│   ├── handlers.py        # API: /process-document, /health
│   ├── main.py            # Инициализация FastAPI-приложения
│   ├── requirements.txt   # Зависимости backend
│   └── task.py            # OCR + LLM, извлечение ФИО
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt   # Зависимости frontend
    └── streamlit_app.py   # Интерфейс загрузки и отображения результата

---

## Развёртывание через Docker (рекомендуется)

### Предварительные требования

- Установлены:
  - Docker
  - Docker Compose (v2.1+)

### Сборка и запуск

Из корня проекта:

docker compose up -d --build

После успешного запуска:

- Frontend (Streamlit) будет доступен по адресу:
  http://localhost:8501/
- Backend (API) — по адресу:
  http://localhost:8000/

На сервере доступ к фронту организован по адресу:
http://45.159.211.61:8501/
- тут по сути к public host'у прокидывается порт 8501(Это от streamlit'a)
