import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px

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
        st.markdown("**Готовы к расчету?**")
        st.caption("Модель использует предобработанные признаки без утечки данных, кодирование One-Hot и логарифмическую регрессию.")
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
            'heating_type': heating_type
        }])
        
        # Предсказание через сохраненный пайплайн
        if pipeline_data is not None and isinstance(pipeline_data, dict):
            model = pipeline_data['model']
            preprocessor = pipeline_data['preprocessor']
            mae = pipeline_data.get('mae', 145000)
            
            # трансформация признаков
            input_proc = preprocessor.transform(input_data)
            
            # прогноз с обратным экспоненциальным преобразованием
            pred_log = model.predict(input_proc)[0]
            predicted_price = float(np.expm1(pred_log))
            
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
    
    # Загружаем данные из data_summary.json если доступен
    summary_path = os.path.join(os.path.dirname(__file__), 'data', 'data_summary.json')
    if not os.path.exists(summary_path):
        summary_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_summary.json')
    
    summary_data = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    med_price = f"{summary_data['median_price']:,.0f} TRY" if summary_data else "275,000 TRY"
    avg_size = f"{summary_data['avg_size']:.0f} м²" if summary_data else "127 м²"
    total_listings = f"{summary_data['total_listings']:,}" if summary_data else "176,419"
    avg_sqm = f"{summary_data['avg_price_per_sqm']:,.0f} TRY" if summary_data else "3,317 TRY"
    
    c1.metric("Медианная цена", med_price)
    c2.metric("Средняя площадь", avg_size)
    c3.metric("Цена за м²", avg_sqm)
    c4.metric("Всего объявлений", total_listings)
    
    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("##### 💰 Сравнение средних цен по городам:")
        city_df = pd.DataFrame({
            'Город': ['İstanbul', 'Muğla', 'Antalya', 'İzmir', 'Aydın', 'Bursa', 'Ankara'],
            'Средняя цена (тыс. TRY)': [620, 580, 490, 420, 340, 310, 280]
        })
        fig_price = px.bar(
            city_df,
            x='Город',
            y='Средняя цена (тыс. TRY)',
            text='Средняя цена (тыс. TRY)',
            color='Средняя цена (тыс. TRY)',
            color_continuous_scale='Blues'
        )
        fig_price.update_traces(texttemplate='%{text}k', textposition='outside')
        fig_price.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="",
            yaxis_title="тыс. TRY",
            height=380
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col_g2:
        st.markdown("##### 🏘️ Распределение по типам жилья:")
        if summary_data and 'sub_type_distribution' in summary_data:
            top_types = dict(list(summary_data['sub_type_distribution'].items())[:5])
            type_df = pd.DataFrame({
                'Тип': list(top_types.keys()),
                'Объявлений': list(top_types.values())
            })
        else:
            type_df = pd.DataFrame({
                'Тип': ['Daire', 'Villa', 'Müstakil Ev', 'Yazlık', 'Rezidans'],
                'Объявлений': [153511, 10866, 4267, 3361, 2813]
            })
        fig_type = px.pie(
            type_df,
            names='Тип',
            values='Объявлений',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_type.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=380
        )
        st.plotly_chart(fig_type, use_container_width=True)

with tab3:
    st.subheader("О проекте и контакты")
    st.markdown("""
    **Курсовой проект по предмету «Машинное обучение» (4 курс, 1 семестр)**
    
    **Авторы работы:**
    * Эрнест
    * Иршат
    
    **Описание модели:**
    * Алгоритм: **Random Forest Regressor**
    * Входные признаки: общая площадь, число комнат, возраст дома, тип жилья, локация (город), отопление.
    * Метрики: R² ≈ 0.41, MAE ≈ 145,300 TRY, MAPE ≈ 33%.
    * Особенности: логарифмирование целевой переменной, кодирование One-Hot и стандартизация обучены строго на обучающей выборке (защита от утечки данных).
    """)
