import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from catboost import CatBoostRegressor
from datetime import date
import time


# تحميل النموذج المدرب
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# إحداثيات المدن ال 12
cities = {
    "Jenin": (32.4613, 35.3009),
    "Tulkarm": (32.3104, 35.0286),
    "Qalqilya": (32.1896, 34.9706),
    "Nablus": (32.2211, 35.2544),
    "Ramallah": (31.9074, 35.2034),
    "Jerusalem": (31.7683, 35.2137),
    "Bethlehem": (31.7054, 35.2024),
    "Hebron": (31.5326, 35.0998),
    "Jericho": (31.8667, 35.4500),
    "Gaza": (31.5017, 34.4668),
    "KhanYunis": (31.3460, 34.3065),
    "Rafah": (31.2972, 34.2436)
}
# اقتران الطقس
def weather_features(user_date):
#تحويل النص المُدخَل لتاريخ
    user_date = pd.to_datetime(
        user_date,
        format="%d/%m/%Y"
    )
# بداية ونهاية الاسبوع السابق
    lag_start = (user_date - timedelta(days=7)).strftime("%Y-%m-%d")
    lag_end   = (user_date - timedelta(days=1)).strftime("%Y-%m-%d")
# بداية ونهاية الاسبوع المتنبأ به
    target_start = user_date.strftime("%Y-%m-%d")
    target_end   = (user_date + timedelta(days=6)).strftime("%Y-%m-%d")
# قائمة البيانات السباقة
    lag_all = []
#قائمة البيانات المتوقعة
    target_all = []
# حلقة المدن
    for city, (lat, lon) in cities.items():
        # تجهيز رابط بيانات الاسبوع السابق
        lag_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}"
            f"&longitude={lon}"
            f"&start_date={lag_start}"
            f"&end_date={lag_end}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia/Jerusalem"
        )
        # طلب بيانات الاسبوع السابق من open metoe
        r1 = requests.get(lag_url, timeout=10)
        # التأكد أن هناك اجابة وليس ايرور 404 مثلاً
        if r1.status_code == 200:
            data = r1.json()  # تحويل البيانات لصيغة تُقرأ
            # التأكد بأنه يوجد بيانات
            if "daily" in data:
                for i in range(len(data["daily"]["time"])):
                    # اضافة وتحويل الحرارة والرطوبة لمتوسط يومي
                    lag_all.append({
                        "Date": data["daily"]["time"][i],
                        "City": city,
                        "Temperature": (
                            data["daily"]["temperature_2m_max"][i]
                            + data["daily"]["temperature_2m_min"][i]
                        ) / 2,
                        "Humidity": (
                            data["daily"]["relative_humidity_2m_max"][i]
                            + data["daily"]["relative_humidity_2m_min"][i]
                        ) / 2
                    })

        # تجهيز رابط البيانات المتوقة
        target_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}"
            f"&longitude={lon}"
            f"&start_date={target_start}"
            f"&end_date={target_end}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia/Jerusalem"
        )
        # طلب البيانات من open metoe
        r2 = requests.get(target_url, timeout=10)
        # التأكد من قبول الطلب
        if r2.status_code == 200:
            data = r2.json()

            if "daily" in data:
                for i in range(len(data["daily"]["time"])):

                    target_all.append({
                        "Date": data["daily"]["time"][i],
                        "City": city,
                        "Temperature": (
                            data["daily"]["temperature_2m_max"][i]
                            + data["daily"]["temperature_2m_min"][i]
                        ) / 2,
                        "Humidity": (
                            data["daily"]["relative_humidity_2m_max"][i]
                            + data["daily"]["relative_humidity_2m_min"][i]
                        ) / 2
                    })

    # تحويل البيانات لجداول
    lag_df = pd.DataFrame(lag_all)
    target_df = pd.DataFrame(target_all)


# 👇 حماية ضد البيانات الفارغة
    if target_df.empty:
        return("⚠️ لا توجد بيانات تنبؤ للتاريخ المطلوب ")

    if lag_df.empty:
        return("⚠️ لا توجد بيانات تاريخية للتاريخ المطلوب")
    #ايجاد متوسط يومي لفلسطين
    lag_city_daily = lag_df.groupby(["Date", "City"])[["Temperature", "Humidity"]].mean().reset_index()
    lag_pal_daily = lag_city_daily.groupby("Date")[["Temperature", "Humidity"]].mean().reset_index()
    # ايجاد متوسط يومي لفلسطين
    target_city_daily = target_df.groupby(["Date", "City"])[["Temperature", "Humidity"]].mean().reset_index()
    target_pal_daily = target_city_daily.groupby("Date")[["Temperature", "Humidity"]].mean().reset_index()
    # ايجاد متوسط اسبوعي لفلسطين
    lag_week = lag_pal_daily.iloc[:7]
    target_week = target_pal_daily.iloc[:7]
    # التأكد من وجود بيانات قبل المتوسط
    Lag_Temp = lag_week["Temperature"].mean() if not lag_week.empty else None
    Lag_Humidity = lag_week["Humidity"].mean() if not lag_week.empty else None

    Temperature = target_week["Temperature"].mean() if not target_week.empty else None
    Humidity = target_week["Humidity"].mean() if not target_week.empty else None
     #ايجاد رقم الاسوع في السنة والسنة
    year = user_date.isocalendar().year
    week_of_year = user_date.isocalendar().week
    # تمرير كل القيم
    return year, week_of_year,float(Temperature),float(Humidity),float(Lag_Temp),float(Lag_Humidity)

#--------------------------------------------------------------------------------------------------
# خوارزمية CatBoost
#--------------------------------------------------------------------------------------------------

def CatBoost_Predict(year, week_of_year, temperature, humidity, lag_temp, lag_humidity):

    # تحويل رقم الأسبوع إلى تمثيل دوري
    week_sin = np.sin(2 * np.pi * week_of_year / 52)
    week_cos = np.cos(2 * np.pi * week_of_year / 52)

    # إنشاء بيانات الإدخال
    input_data = pd.DataFrame({
        "Year": [year],
        "WeekOfYear": [week_of_year],
        "Week_Sin": [week_sin],
        "Week_Cos": [week_cos],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Temp_Lag1": [lag_temp],
        "Humidity_Lag1": [lag_humidity]
    })

    # التنبؤ
    prediction = model.predict(input_data)[0]

    return round(prediction)

#--------------------------------------------------------------------------------------------------
# واجهة المستخدم
#--------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Disease Prediction", layout="wide")

# ===== CSS =====
st.markdown("""
<style>

/* خلفية عامة */
.stApp {
    background-color: white;
}

/* ===== العنوان ===== */
.title {
    text-align: center;
    color: #4BB8FA;
    font-size: 42px;
    font-weight: 800;
    margin-top: -10px;
    line-height: 1.3;
}

/* ===== date input ===== */
div[data-testid="stDateInput"] {
    width: 420px;
    margin: auto;
}

div[data-testid="stDateInput"] label {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #4BB8FA !important;
    text-align: center;
}

div[data-testid="stDateInput"] input {
    font-size: 18px !important;
    padding: 14px !important;
    height: 55px !important;
    border: 2px solid #A7D8FF !important;
    border-radius: 10px !important;
    background-color: #F2FAFF !important;
}

/* ===== زر predict ===== */
div.stButton > button {
    background-color: #F7A8C4;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 30px;
    border-radius: 12px;
    border: none;
    display: block;
    margin: 20px auto;
}

div.stButton > button:hover {
    background-color: #f28bb2;
}

/* ===== risk النص ===== */
.risk-box {
    text-align: right;
    font-size: 22px;
    font-weight: 700;
    color: #222;
    padding: 15px;
    border-right: 5px solid #4BB8FA;
    margin-top: 20px;
}

/* message */
.msg-box {
    text-align: right;
    font-size: 18px;
    color: #444;
    line-height: 1.7;
    margin-top: 10px;
}
div.stButton > button {
    background-color: #F7A8C4;
    color: white;
    font-size: 22px !important;
    font-weight: 800;
    padding: 14px 50px;
    border-radius: 14px;
    border: none;
    width: 100%;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #f28bb2;
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
left, center, right = st.columns([1, 6, 1])

with center:
    st.markdown("""
    <div style="
        text-align:center;
        color:#4BB8FA;
        font-size:42px;
        font-weight:800;
        line-height:1.4;
        margin-top:10px;">
        Digital Palestine Immunity<br>
        مناعة فلسطين الرقمية
    </div>
    """, unsafe_allow_html=True)

with right:
    st.image("logo.png", width=120)


# ===== INPUT =====
col, coll, colll = st.columns([1, 1, 1])
with coll:
    selected_date = st.date_input("Select Week Start Date | اختر بداية الأسبوع", width=120)
tomorrow = date.today() + timedelta(days=1)

if selected_date > tomorrow:
    st.error("There's no data for this date.")
    st.stop()
# ===== BUTTON =====
st.markdown("""
<style>
div.stButton > button {
    width: 200px;
    height: 50px;
    border-radius: 12px;
}

/* تكبير النص داخل الزر */
div.stButton > button * {
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* صندوق عدد الإصابات */
.cases-box{
    width:550px;
    margin:25px auto 15px auto;
    padding:20px;
    background:#E8F5E9;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:700;
    color:#0B6623;
    box-shadow:0 4px 12px rgba(0,0,0,.15);
    direction: rtl;
    text-align: right;
}

/* صندوق الخطورة */
.risk-box{
    width:550px;
    margin:20px auto 12px auto;
    padding:18px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:700;
    box-shadow:0 4px 12px rgba(0,0,0,.12);
    direction: rtl;
    text-align: right;
}

/* صندوق الرسالة */
.msg-box{
    width:550px;
    margin:0 auto 20px auto;
    padding:20px;
    border-radius:15px;
    text-align:justify;
    line-height:1.9;
    font-size:18px;
    font-weight:500;
    box-shadow:0 4px 12px rgba(0,0,0,.10);
    direction: rtl;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.4, 1, 1])
with col2:
    predict = st.button("Predict | تنبؤ")

if predict:
    placeholder = st.empty()

    placeholder.markdown("""
    <div style="text-align:center; color:#4BB8FA; font-weight:800; font-size:18px;">
    📥 جارٍ تجهيز البيانات
    </div>
    """, unsafe_allow_html=True)

    year, week_of_year, temperature, humidity, lag_temp, lag_humidity = weather_features(selected_date)
    time.sleep(0.3)
    placeholder.markdown("""
    <div style="text-align:center; color:#4BB8FA; font-weight:800; font-size:18px;">
    🧠 جارٍ تشغيل نموذج التنبؤ
    </div>
    """, unsafe_allow_html=True)

    cases = CatBoost_Predict(year, week_of_year, temperature, humidity, lag_temp, humidity)
    time.sleep(0.3)
    placeholder.markdown("""
    <div style="text-align:center; color:#4BB8FA; font-weight:800; font-size:18px;">
    📊 جارٍ تحليل النتائج
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.3)

    placeholder.markdown("""
    <div style="text-align:center; color:#4BB8FA; font-weight:800; font-size:18px;">
    ✅ تم التنبؤ بنجاح
    </div>
    """, unsafe_allow_html=True)
    # Risk logic
    if cases <= 10:
        risk = "🟢 المستوى منخفض"
        risk_bg = "#E8F5E9"
        risk_color = "#1B5E20"
        msg_bg = "#F4FBF4"
        msg = """الوضع مستقر وآمن بشكل عام؛ حيث إن عدد الإصابات المتوقعة يقع ضمن معدلاته المنخفضة والطبيعية، ولا يوجد ما يدعو للقلق، مع الاستمرار في اتباع تدابير الوقاية الشخصية الاعتيادية."""
    elif cases <= 18:
        risk = "🟡 المستوى متوسط"
        risk_bg = "#FFF8E1"
        risk_color = "#8D6E00"
        msg_bg = "#FFFDF3"
        msg = """تشير التوقعات إلى ارتفاع تدريجي في النشاط الموسمي للفيروس نتيجة تغير الظروف الجوية، ويُنصح بأخذ لقاح الإنفلونزا السنوي، وتوفير مستلزمات الوقاية الأساسية، ومراجعة الطبيب عند ظهور الأعراض الأولية."""
    elif cases <= 24:
        risk = "🟠 المستوى مرتفع"
        risk_bg = "#FFF3E0"
        risk_color = "#E65100"
        msg_bg = "#FFF8F2"
        msg = """تشير المؤشرات إلى موجة انتشار موسمية واضحة، لذلك يُوصى بارتداء الكمامات في المرافق الصحية، وتجنب الأماكن المغلقة والمزدحمة، والحرص على توفير الأدوية والالتزام بالإجراءات الوقائية."""
    else:
        risk = "🔴 المستوى خطير جداً"
        risk_bg = "#FDECEC"
        risk_color = "#B71C1C"
        msg_bg = "#FFF5F5"
        msg = """تحذير عاجل! تشير التوقعات إلى وصول الإصابات إلى مستويات مرتفعة جداً، لذا يجب رفع الجاهزية الطبية، والالتزام الكامل بإجراءات الوقاية، والتوجه للمستشفى فور ظهور أي مضاعفات تنفسية شديدة."""
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
    <div class="cases-box">
        Predicted Cases | عدد الإصابات المتوقعة : {cases}
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
    <div class="risk-box"
    style="
    background:{risk_bg};
    color:{risk_color};
    ">
    {risk}
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
    <div class="msg-box"
    style="
    background:{msg_bg};
    color:#333333;
    ">
    {msg}
    </div>
    """, unsafe_allow_html=True)











