import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Zingat Real Estate - Прогноз стоимости",
    page_icon="🏠",
    layout="wide"
)

# Загрузка обученной модели и препроцессоров
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_pipeline.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_pipeline.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline_data = load_pipeline()

st.title("🏠 Оценка рыночной стоимости недвижимости (Zingat)")

# Навигация по вкладкам
tab1, tab2, tab3 = st.tabs(["📊 Калькулятор стоимости", "📈 Аналитический дашборд", "ℹ️ Справка и авторы"])

with tab1:
    st.subheader("Введите параметры объекта недвижимости:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox(
            "Город:",
            ["İstanbul", "Ankara", "İzmir", "Antalya", "Aydın", "Bursa", "Muğla", "Diger"]
        )
        sub_type = st.selectbox(
            "Тип недвижимости:",
            ["Daire", "Villa", "Rezidans", "Müstakil Ev", "Yazlık", "Komple Bina"]
        )
        size = st.number_input("Общая площадь (м²):", min_value=20.0, max_value=400.0, value=100.0, step=5.0)

    with col2:
        total_rooms = st.selectbox(
            "Количество комнат:",
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            format_func=lambda x: f"{x} комнаты (например, {x-1}+1)"
        )
        building_age_num = st.selectbox(
            "Возраст здания (лет):",
            options=[0, 1, 2, 3, 4, 5, 8, 13, 18, 23, 28, 35],
            index=2,
            format_func=lambda x: "Новостройка (0 лет)" if x == 0 else f"{x} лет"
        )
        heating_type = st.selectbox(
            "Тип отопления:",
            ["Kombi (Doğalgaz)", "Merkezi Sistem", "Klima", "Yerden Isıtma", "Yok", "Soba (Kömür)"]
        )

    with col3:
        tom = st.slider("Дней на рынке (Time on Market):", min_value=1, max_value=365, value=30)
        st.markdown("<br>", unsafe_allow_html=True)
        calc_button = st.button("🚀 Рассчитать стоимость", use_container_width=True, type="primary")

    if calc_button:
        # Формируем датафрейм признаков
        input_data = pd.DataFrame([{
            'size': size,
            'total_rooms': total_rooms,
            'building_age_num': building_age_num,
            'sub_type': sub_type,
            'city': city,
            'heating_type': heating_type,
            'tom': tom
        }])
        
        # Предсказание
        if pipeline_data is not None and isinstance(pipeline_data, dict):
            model = pipeline_data['model']
            encoder = pipeline_data['encoder']
            cat_cols = pipeline_data['cat_cols']
            
            # кодируем категории
            input_encoded = input_data.copy()
            input_encoded[cat_cols] = encoder.transform(input_encoded[cat_cols].astype(str))
            
            predicted_price = model.predict(input_encoded)[0]
            mae = 125000  # средняя ошибка модели
            
            st.success(f"### Прогнозируемая цена: **{predicted_price:,.0f} TRY** (~ {predicted_price/35:,.0f} $)")
            st.info(f"Доверительный диапазон (±MAE): от **{max(20000, predicted_price - mae):,.0f}** до **{predicted_price + mae:,.0f} TRY**")
        else:
            # Демо-расчет если модель еще не обучена
            base_sqm = 4500 if city == "İstanbul" else 3000
            demo_price = size * base_sqm * (1 + 0.1 * total_rooms)
            st.success(f"### Примерная оценка: **{demo_price:,.0f} TRY**")
            st.caption("(Модель model_pipeline.joblib пока не найдена в папке models/)")

with tab2:
    st.subheader("Статистика по рынку недвижимости (Zingat)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Медианная цена", "275,000 TRY")
    c2.metric("Средняя площадь", "127 м²")
    c3.metric("Доля квартир", "87%")
    
    st.markdown("---")
    st.write("#### Сравнение средних цен по городам:")
    city_chart = pd.DataFrame({
        'Город': ['İstanbul', 'Muğla', 'Antalya', 'İzmir', 'Aydın', 'Bursa', 'Ankara'],
        'Средняя цена (тыс. TRY)': [620, 580, 490, 420, 340, 310, 280]
    }).set_index('Город')
    st.bar_chart(city_chart)

with tab3:
    st.subheader("О проекте и контакты")
    st.markdown("""
    **Курсовой проект по предмету «Машинное обучение» (4 курс, 1 семестр)**
    
    **Авторы работы:**
    * Эрнест
    * Иршат
    
    **Описание модели:**
    * Алгоритм: **Random Forest Regressor**
    * Входные признаки: общая площадь, число комнат, возраст дома, тип жилья, локация, отопление, время на рынке.
    * Метрики: R² ≈ 0.61, MAE ≈ 122,850 TRY.
    * Особенности: кодирование категорий и масштабирование обучены строго на обучающей выборке (защита от утечки данных).
    """)
