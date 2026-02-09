import streamlit as st
import requests
from PIL import Image

st.title("Извлечение ФИО из документа")

uploaded_file = st.file_uploader("Загрузите фото документа", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Показываем загруженное изображение
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженный документ", use_container_width=True)

    if st.button("Извлечь ФИО"):
        with st.spinner("Обработка..."):
            uploaded_file.seek(0)
            response = requests.post(
                "http://backend:8000/process-document",
                files={"file": (uploaded_file.name, uploaded_file)}
            )

            if response.status_code == 200:
                fio = response.json()["fio"]
                st.success(f"**ФИО:** {fio}")
            else:
                st.error(f"Ошибка: {response.text}")
