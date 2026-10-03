import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Food Quality Checker", page_icon="🥗")

st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #2E7D32; text-align: center; }
    .sub-title { font-size: 16px; color: #555; margin-bottom: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# กลับมาใช้คำสั่งโหลดโมเดลของ Streamlit 1.11.0
@st.experimental_singleton
def load_model():
    return tf.keras.models.load_model('food_freshness_model.keras')

model = load_model()

class_names = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 
    'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 
    'freshtomato', 'rottenapples', 'rottenbanana', 'rottenbittergroud', 
    'rottencapsicum', 'rottencucumber', 'rottenokra', 'rottenoranges', 
    'rottenpotato', 'rottentomato'
]

class_translator = {
    'freshapples': ('แอปเปิล 🍎', 'สด ✨', 'success', '14 - 21 วัน'),
    'freshbanana': ('กล้วย 🍌', 'สด ✨', 'success', '3 - 5 วัน (เปลือกอาจคล้ำขึ้น)'),
    'freshbittergroud': ('มะระ 🥒', 'สด ✨', 'success', '3 - 5 วัน'),
    'freshcapsicum': ('พริกหยวก 🫑', 'สด ✨', 'success', '5 - 7 วัน'),
    'freshcucumber': ('แตงกวา 🥒', 'สด ✨', 'success', '5 - 7 วัน'),
    'freshokra': ('กระเจี๊ยบเขียว 🌿', 'สด ✨', 'success', '3 - 5 วัน'),
    'freshoranges': ('ส้ม 🍊', 'สด ✨', 'success', '14 - 21 วัน'),
    'freshpotato': ('มันฝรั่ง 🥔', 'สด ✨', 'success', 'ไม่แนะนำให้แช่ตู้เย็น (เก็บอุณหภูมิห้องได้ 1-2 เดือน)'),
    'freshtomato': ('มะเขือเทศ 🍅', 'สด ✨', 'success', '3 - 5 วัน (แช่เย็นอาจทำให้เสียรสชาติ)'),
    
    'rottenapples': ('แอปเปิล 🍎', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottenbanana': ('กล้วย 🍌', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottenbittergroud': ('มะระ 🥒', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottencapsicum': ('พริกหยวก 🫑', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottencucumber': ('แตงกวา 🥒', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottenokra': ('กระเจี๊ยบเขียว 🌿', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottenoranges': ('ส้ม 🍊', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottenpotato': ('มันฝรั่ง 🥔', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)'),
    'rottentomato': ('มะเขือเทศ 🍅', 'เน่าเสีย ⚠️', 'error', 'หมดอายุ (ทิ้งทันที)')
}

st.markdown('<div class="main-title">🥗 ระบบประเมินคุณภาพความสดของผักผลไม้ด้วย AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ตรวจสอบความสดของอาหารได้ทันที ทั้งแบบอัปโหลดภาพและสแกนผ่านกล้อง</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📂 โหมดอัปโหลดรูปภาพ", "📷 โหมดสแกนเรียลไทม์ (กล้อง)"])

# ================= TAB 1 =================
with tab1:
    uploaded_file = st.file_uploader("ลากไฟล์ หรือคลิกเพื่ออัปโหลดรูปภาพ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='รูปภาพที่กำลังวิเคราะห์', use_column_width=True)
            
        with col2:
            st.subheader("📊 ผลการวิเคราะห์")
            with st.spinner("AI กำลังสแกนรูปภาพ..."):
                img_resized = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = predictions[0]
                
                predicted_class_raw = class_names[np.argmax(score)]
                confidence = float(np.max(score))
                
                food_name, status, alert_type, shelf_life = class_translator[predicted_class_raw]
                
                st.metric(label="ตรวจพบหมวดหมู่", value=food_name)
                if alert_type == 'success':
                    st.success(f"**สถานะ:** {status}")
                    st.info(f"❄ **ระยะเวลาเก็บรักษาในตู้เย็น:** {shelf_life}")
                else:
                    st.error(f"**สถานะ:** {status}")
                    st.warning(f"🗑 **คำแนะนำ:** {shelf_life}")
                    
                st.write("---")
                st.write(f"**ความมั่นใจของ AI:** {confidence * 100:.2f}%")
                st.progress(confidence)

# ================= TAB 2 =================
with tab2:
    st.write("กดปุ่ม **START** ด้านล่างเพื่อเปิดกล้องคอมพิวเตอร์ของคุณ แล้วนำผักหรือผลไม้มาจ่อหน้ากล้องได้เลย")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        confidence = float(np.max(score))
        
        if confidence > 0.70:
            predicted_class_raw = class_names[np.argmax(score)]
            
            status_eng = "FRESH" if "fresh" in predicted_class_raw else "ROTTEN"
            item_name = predicted_class_raw.replace("fresh", "").replace("rotten", "").capitalize()
            
            label = f"{item_name}: {status_eng} ({confidence*100:.1f}%)"
            color = (0, 255, 0) if status_eng == "FRESH" else (0, 0, 255)
            
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="food-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    st.info("💡 หมายเหตุ: ข้อความบนกล้องจะเป็นภาษาอังกฤษเนื่องจาก OpenCV ไม่รองรับฟอนต์ไทยในตัวครับ")