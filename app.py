import streamlit as st
import cv2
import numpy as np
import easyocr
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import pandas as pd


# =====================
# LOAD DATABASE FROM EXCEL
# =====================
@st.cache_data
def load_database():
    try:
        # Read the Excel file
        df = pd.read_excel('ghana_dvla_dummy_data.xlsx')

        # Convert to dictionary format
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
                # Adding some dummy status fields based on registration date
                "status": "VALID" if pd.to_datetime(row['Registration Date']) > pd.to_datetime(
                    '2020-01-01') else "EXPIRED",
                "insurance": "Active" if pd.to_datetime(row['Registration Date']) > pd.to_datetime(
                    '2020-01-01') else "Expired",
                "stolen": False  # Default to not stolen
            }

        # Mark some plates as stolen for demonstration
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
# IMAGE PROCESSING
# =====================
def detect_plate(image):
    model = load_yolo()
    if model is None:
        return None

    # Convert BGR to RGB and run detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, conf=0.5)  # Minimum 50% confidence

    if len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # Draw debug rectangle
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
        results = reader.readtext(thresh, detail=0)
        return " ".join(results).strip() if results else None
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None


# =====================
# WEBCAM FUNCTIONS
# =====================
def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def stop_webcam(cap):
    if cap is not None:
        cap.release()


def capture_frame(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    return None


# =====================
# STREAMLIT UI
# =====================
st.set_page_config(
    page_title="DVLA Plate Check",
    layout="wide",
    page_icon="🚔"
)

st.title("🇬🇭 Ghana DVLA & Police Plate Verification")
st.markdown("---")

# User Type Selection
user_type = st.radio("Login As:", ["DVLA Officer", "Police Officer"], horizontal=True)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Live Camera", "📁 Upload Image", "🔍 Manual Check"])

with tab1:
    st.subheader("Live Camera Feed")

    # Initialize session state for webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎥 Start Webcam"):
            st.session_state.cap = start_webcam()
            st.session_state.webcam_active = True
            st.success("Webcam activated!")

        if st.button("🛑 Stop Webcam"):
            if st.session_state.cap is not None:
                stop_webcam(st.session_state.cap)
                st.session_state.webcam_active = False
                st.session_state.cap = None
                st.warning("Webcam stopped")

    if st.session_state.webcam_active and st.session_state.cap is not None:
        frame_placeholder = st.empty()
        capture_button = st.button("📸 Capture & Process")

        while st.session_state.webcam_active:
            frame = capture_frame(st.session_state.cap)
            if frame is not None:
                frame_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")

            if capture_button:
                if frame is not None:
                    with st.spinner("Processing captured image..."):
                        plate_img = detect_plate(frame)

                        if plate_img is not None:
                            plate_text = read_plate(plate_img)

                            if plate_text:
                                plate_text = plate_text.upper()
                                st.success(f"Detected Plate: **{plate_text}**")
                                st.image(plate_img, caption="Detected License Plate", use_container_width=True)

                                # Database Check
                                plate_data = DUMMY_DB.get(plate_text, None)

                                with col2:
                                    st.subheader("📋 Registration Details")

                                    if plate_data:
                                        # Display validation status prominently
                                        if plate_data["status"] == "VALID":
                                            st.success("✅ VALID LICENSE PLATE")
                                        elif plate_data["status"] == "STOLEN":
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
                                            if plate_data["status"] == "STOLEN":
                                                st.button("🚨 Alert All Units", type="primary")
                                            elif plate_data["status"] == "VALID":
                                                st.success("✅ Insurance is ACTIVE")
                                            else:
                                                st.error("❌ Insurance EXPIRED")
                                                st.button("Issue Citation", type="primary")
                                    else:
                                        st.warning("⚠️ Plate not found in database")
                                        if user_type == "DVLA Officer":
                                            if st.button("➕ Add New Registration"):
                                                st.session_state.new_plate = plate_text
                                                st.experimental_rerun()
                            else:
                                st.error("Could not read plate text")
                        else:
                            st.error("No license plate detected")
                break

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
                    with st.spinner("Reading plate number..."):
                        plate_text = read_plate(plate_img)

                    if plate_text:
                        plate_text = plate_text.upper()
                        st.success(f"Detected Plate: **{plate_text}**")
                        st.image(plate_img, caption="Detected License Plate", use_container_width=True)

                        # Database Check
                        plate_data = DUMMY_DB.get(plate_text, None)

                        with col2:
                            st.subheader("📋 Registration Details")

                            if plate_data:
                                # Display validation status prominently
                                if plate_data["status"] == "VALID":
                                    st.success("✅ VALID LICENSE PLATE")
                                elif plate_data["status"] == "STOLEN":
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
                                    if plate_data["status"] == "STOLEN":
                                        st.button("🚨 Alert All Units", type="primary")
                                    elif plate_data["status"] == "VALID":
                                        st.success("✅ Insurance is ACTIVE")
                                    else:
                                        st.error("❌ Insurance EXPIRED")
                                        st.button("Issue Citation", type="primary")
                            else:
                                st.warning("⚠️ Plate not found in database")
                                if user_type == "DVLA Officer":
                                    if st.button("➕ Add New Registration"):
                                        st.session_state.new_plate = plate_text
                                        st.experimental_rerun()
                    else:
                        st.error("Could not read plate text")
                else:
                    st.error("No license plate detected")

with tab3:
    st.subheader("Manual Plate Check")
    plate_input = st.text_input("Enter Plate Number (e.g. GA4051-24):").upper()
    if plate_input:
        plate_data = DUMMY_DB.get(plate_input, None)
        if plate_data:
            # Display validation status for manual check
            if plate_data["status"] == "VALID":
                st.success("✅ VALID LICENSE PLATE")
            elif plate_data["status"] == "STOLEN":
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
                if plate_data["status"] == "STOLEN":
                    st.button("🚨 Alert All Units", type="primary")
                elif plate_data["status"] == "VALID":
                    st.success("✅ Insurance is ACTIVE")
                else:
                    st.error("❌ Insurance EXPIRED")
                    st.button("Issue Citation", type="primary")

        else:
            st.warning("Plate not found in database")
            if user_type == "DVLA Officer":
                if st.button("➕ Add New Registration"):
                    st.session_state.new_plate = plate_input
                    st.experimental_rerun()

# Footer
st.markdown("---")
st.caption(f"Ghana DVLA & Police System | {datetime.now().year} | Developed Ezer-Tech")