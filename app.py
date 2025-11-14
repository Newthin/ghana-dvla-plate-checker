import streamlit as st
import cv2
import numpy as np
import easyocr
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import re

# === PWA: CORE SUPPORT (Manifest + Icons + Service Worker) ===
st.set_page_config(
    page_title="DVLA Scan",
    page_icon="icon-192.png",
    layout="wide",
    menu_items=None
)

st.markdown("""
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon-192.png" type="image/png">
<meta name="theme-color" content="#1E90FF">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
""", unsafe_allow_html=True)

# Register service worker (HTML only, no script — avoids React error)
st.markdown("""
<script src="/service-worker.js"></script>
""", unsafe_allow_html=True)

# =====================
# ZOOMABLE IMAGE FUNCTION
# =====================
def zoomable_image(image, caption="", zoom=True):
    """Show image with click-to-zoom (expander)"""
    if zoom:
        with st.expander(f"Zoom: {caption}", expanded=False):
            st.image(image, use_container_width=True, channels="BGR")
    else:
        st.image(image, caption=caption, use_container_width=True, channels="BGR")

# =====================
# LOAD DATABASE FROM EXCEL
# =====================
@st.cache_data
def load_database():
    try:
        df = pd.read_excel('ghana_dvla_dummy_data.xlsx')
        db = {}
        for _, row in df.iterrows():
            plate = row['License Plate']
            db[plate] = {
                "owner": row['Owner Name'],
                "make": row['Make'],
                "model": row['Model'],
                "year": row['Year of Manufacture'],
                "color": row['Color'],
                "registration_date": row['Registration Date'],
                "status": "VALID" if pd.to_datetime(row['Registration Date']) > pd.to_datetime('2020-01-01') else "EXPIRED",
                "insurance": "Active" if pd.to_datetime(row['Registration Date']) > pd.to_datetime('2020-01-01') else "Expired",
                "stolen": False
            }
        stolen_plates = ["GA4051-24", "BE0607-22", "AS3089-14"]
        for plate in stolen_plates:
            if plate in db:
                db[plate]["status"] = "STOLEN"
                db[plate]["stolen"] = True
                db[plate]["insurance"] = "Suspended"
        return db
    except Exception as e:
        st.error(f"Failed to load database: {str(e)}")
        return {}

DUMMY_DB = load_database()

# =====================
# LOAD MODELS (CACHED)
# =====================
@st.cache_resource
def load_yolo():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_ocr():
    try:
        return easyocr.Reader(['en'])
    except Exception as e:
        st.error(f"OCR loading failed: {str(e)}")
        return None

# =====================
# CLEAN GHANA PLATE TEXT (FIXES GH + O/0)
# =====================
def clean_ghana_plate(ocr_text: str) -> str | None:
    if not ocr_text:
        return None
    text = re.sub(r'[^A-Za-z0-9]', '', ocr_text.upper())
    if text.startswith('GH'):
        text = text[2:]
    if text.startswith(('O', '0')):
        text = text[1:]
    m = re.match(r'^([A-Z]{2})(\d{4})(\d{2})$', text)
    if m:
        region, num, year = m.groups()
        return f"{region} {num}-{year}"
    m = re.match(r'^([A-Z]{2})(\d{3})(\d{2})$', text)
    if m:
        region, num, year = m.groups()
        return f"{region} {num}-{year}"
    m = re.match(r'^(G[XC])(\d{4})$', text)
    if m:
        code, num = m.groups()
        return f"{code} {num}"
    m = re.match(r'^([A-Z]{2})(\d+)', text)
    if m and len(m.group(2)) >= 3:
        region, num = m.groups()
        if len(num) == 6:
            return f"{region} {num[:4]}-{num[4:]}"
        if len(num) == 5:
            return f"{region} {num[:3]}-{num[3:]}"
        return f"{region} {num}"
    return None

# =====================
# IMAGE PROCESSING
# =====================
def detect_plate(image):
    model = load_yolo()
    if model is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, conf=0.5)
    if len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        st.image(debug_img, channels="BGR", caption="Detection Preview", use_container_width=True)
        return image[y1:y2, x1:x2]
    return None

def read_plate(plate_img):
    reader = load_ocr()
    if reader is None:
        return None
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.equalizeHist(thresh)
        results = reader.readtext(thresh, detail=0, paragraph=False)
        raw_text = " ".join(results).strip()
        cleaned = clean_ghana_plate(raw_text)
        if cleaned:
            return cleaned
        else:
            st.warning("Could not parse plate format")
            return None
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

# =====================
# STREAMLIT UI
# =====================
st.title("🇬🇭 Ghana DVLA & Police Plate Verification")
st.markdown("---**Install: Open in Chrome → 3-dots → Add to Home Screen**")

user_type = st.radio("Login As:", ["DVLA Officer", "Police Officer"], horizontal=True)

tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload Image", "🔍 Manual Check"])

# ------------------- TAB 1: CAMERA -------------------
with tab1:
    st.subheader("Live Camera")
    picture = st.camera_input(
        "Point camera at license plate & click **Take Photo**",
        help="Works on mobile & desktop"
    )
    if picture:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Detecting plate..."):
            plate_img = detect_plate(image)
        if plate_img is not None:
            with st.spinner("Reading plate..."):
                plate_text = read_plate(plate_img)
            if plate_text:
                st.success(f"Detected Plate: **{plate_text}**")
                zoomable_image(plate_img, caption="Detected License Plate", zoom=True)
                plate_data = DUMMY_DB.get(plate_text, None)
                if plate_data:
                    status = plate_data["status"]
                    if status == "VALID":
                        st.success("✅ VALID LICENSE PLATE")
                    elif status == "STOLEN":
                        st.error("🚨 STOLEN VEHICLE")
                    else:
                        st.warning("⚠️ EXPIRED LICENSE")
                    st.markdown(f"""
                    - **Owner**: {plate_data['owner']}
                    - **Vehicle**: {plate_data['make']} {plate_data['model']} ({plate_data['year']})
                    - **Color**: {plate_data['color']}
                    - **Registration Date**: {plate_data['registration_date']}
                    - **Insurance**: {plate_data['insurance']}
                    """)
                    if user_type == "Police Officer":
                        if status == "STOLEN":
                            st.button("🚨 Alert All Units", type="primary")
                        elif status == "VALID":
                            st.success("✅ Insurance ACTIVE")
                        else:
                            st.error("❌ Insurance EXPIRED")
                            st.button("Issue Citation", type="primary")
                else:
                    st.warning("Plate not in database")
            else:
                st.error("Could not read plate text")
        else:
            st.error("No license plate detected")

# ------------------- TAB 2: UPLOAD IMAGE -------------------
with tab2:
    st.subheader("Upload Vehicle Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with col1:
            st.image(image, channels="BGR", caption="Uploaded Vehicle", use_container_width=True)
            if st.button("🚔 Scan License Plate"):
                with st.spinner("Detecting plate..."):
                    plate_img = detect_plate(image)
                if plate_img is not None:
                    with st.spinner("Reading plate..."):
                        plate_text = read_plate(plate_img)
                    if plate_text:
                        st.success(f"Detected Plate: **{plate_text}**")
                        zoomable_image(plate_img, caption="Detected License Plate", zoom=True)
                        plate_data = DUMMY_DB.get(plate_text, None)
                        with col2:
                            st.subheader("📋 Registration Details")
                            if plate_data:
                                status = plate_data["status"]
                                if status == "VALID":
                                    st.success("✅ VALID LICENSE PLATE")
                                elif status == "STOLEN":
                                    st.error("🚨 STOLEN VEHICLE")
                                else:
                                    st.warning("⚠️ EXPIRED LICENSE")
                                st.markdown(f"""
                                - **Owner**: {plate_data['owner']}
                                - **Vehicle**: {plate_data['make']} {plate_data['model']} ({plate_data['year']})
                                - **Color**: {plate_data['color']}
                                - **Registration Date**: {plate_data['registration_date']}
                                - **Insurance**: {plate_data['insurance']}
                                """)
                                if user_type == "Police Officer":
                                    if status == "STOLEN":
                                        st.button("🚨 Alert All Units", type="primary")
                                    elif status == "VALID":
                                        st.success("✅ Insurance ACTIVE")
                                    else:
                                        st.error("❌ Insurance EXPIRED")
                                        st.button("Issue Citation", type="primary")
                            else:
                                st.warning("Plate not in database")
                                if user_type == "DVLA Officer":
                                    if st.button("➕ Add New Registration"):
                                        st.session_state.new_plate = plate_text
                                        st.rerun()
                    else:
                        st.error("Could not read plate text")
                else:
                    st.error("No license plate detected")

# ------------------- TAB 3: MANUAL CHECK -------------------
with tab3:
    st.subheader("Manual Plate Check")
    plate_input = st.text_input("Enter Plate Number (e.g. GE 351-19):").upper().strip()
    if plate_input:
        plate_data = DUMMY_DB.get(plate_input, None)
        if plate_data:
            status = plate_data["status"]
            if status == "VALID":
                st.success("✅ VALID LICENSE PLATE")
            elif status == "STOLEN":
                st.error("🚨 STOLEN VEHICLE")
            else:
                st.warning("⚠️ EXPIRED LICENSE")
            st.markdown(f"""
            - **Owner**: {plate_data['owner']}
            - **Vehicle**: {plate_data['make']} {plate_data['model']} ({plate_data['year']})
            - **Color**: {plate_data['color']}
            - **Registration Date**: {plate_data['registration_date']}
            - **Insurance**: {plate_data['insurance']}
            """)
            if user_type == "Police Officer":
                if status == "STOLEN":
                    st.button("🚨 Alert All Units", type="primary")
                elif status == "VALID":
                    st.success("✅ Insurance ACTIVE")
                else:
                    st.error("❌ Insurance EXPIRED")
                    st.button("Issue Citation", type="primary")
        else:
            st.warning("Plate not in database")
            if user_type == "DVLA Officer":
                if st.button("➕ Add New Registration"):
                    st.session_state.new_plate = plate_input
                    st.rerun()

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption(f"🇬🇭 Ghana DVLA & Police System | {datetime.now().year} | Developed by Ezer-Tech")
