import streamlit as st
import cv2
import numpy as np
import sqlite3
from deepface import DeepFace
from ultralytics import YOLO
import uuid

# 初始化 SQLite 資料庫
conn = sqlite3.connect('face_recognition.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY, gender TEXT, average_age REAL)''')
conn.commit()

# 加載 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 確保這個模型能夠檢測帽子和口罩

st.title("即時影像辨識應用 - 性別與年齡辨識")

# 使用按鈕來控制相機
col1, col2 = st.columns(2)
with col1:
    start_button = st.button('啟用相機', key='start')
with col2:
    stop_button = st.button('停止相機', key='stop')

run = False

if start_button:
    with st.spinner('啟用相機中...'):
        run = True
if stop_button:
    with st.spinner('停止相機中...'):
        run = False

if run:
    # 使用 OpenCV 開啟相機
    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    analysis_placeholder = st.sidebar.empty()  # 將分析結果顯示到左側
    id_input_placeholder = st.sidebar.empty()  # 用於顯示或輸入ID
    average_age_placeholder = st.sidebar.empty()  # 顯示年齡平均值
    save_button_placeholder = st.sidebar.empty()  # 用於顯示記錄按鈕
    status_placeholder = st.sidebar.empty()  # 用於顯示狀態

    age_records = []
    gender = None  # 用於保存檢測到的性別

    unique_id = str(uuid.uuid4())

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("無法讀取相機影像")
            break

        # 使用 YOLOv8 進行物件偵測
        results = model(frame)
        
        # 顯示 YOLOv8 狀態
        orig_shape = results[0].orig_shape
        detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
        person_count = detected_objects.count('person')
        hat_count = detected_objects.count('hat') if 'hat' in model.names.values() else 0
        mask_count = detected_objects.count('mask') if 'mask' in model.names.values() else 0
        status_text = f"{orig_shape[1]}x{orig_shape[0]}, {person_count} person(s), {hat_count} hat(s), {mask_count} mask(s), {results[0].speed['inference']}ms\n"
        status_text += f"Speed: {results[0].speed['preprocess']}ms preprocess, {results[0].speed['inference']}ms inference, {results[0].speed['postprocess']}ms postprocess per image at shape {orig_shape}\n"
        status_text += f"Detected objects: {', '.join(detected_objects)}"
        status_placeholder.markdown(f"### 狀態\n{status_text}")

        detected_faces = []
        analysis_results = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = model.names[int(box.cls)]
                if cls_name == 'person':
                    face = frame[y1:y2, x1:x2]

                    # 進行影像分析
                    try:
                        analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

                        for result in analysis:
                            age = result['age']
                            gender = result['dominant_gender']
                            detected_faces.append((gender, age, x1, y1, x2-x1, y2-y1))
                            analysis_results.append(f"Gender: {gender}, Age: {age}")
                            age_records.append(age)

                    except Exception as e:
                        st.write("影像分析失敗：", e)

                # 繪製檢測到的物體框和標籤
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 繪製檢測到的人臉框和標籤
        for (gender, age, x, y, w, h) in detected_faces:
            # 繪製檢測到的人臉框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 繪製性別和年齡在框內
            cv2.putText(frame, f"Gender: {gender}, Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 顯示影像
        stframe.image(frame, channels='BGR')

        # 在左側顯示分析結果
        if analysis_results:
            analysis_placeholder.markdown("### 影像分析結果")
            for result in analysis_results:
                analysis_placeholder.write(result)

        # 計算年齡平均值並顯示
        if age_records:
            average_age = sum(age_records) / len(age_records)
            average_age_placeholder.markdown(f"### 年齡平均值: {average_age:.2f}")

        # 為每個檢測到的個體生成唯一 ID
        id_input_placeholder.markdown(f"### 目前個體的ID: {unique_id}")

    # 提供輸入框輸入人名並存入資料庫
    if save_button_placeholder.button("記錄", key=f'save_button_{unique_id}'):
        c.execute("INSERT INTO people (id, gender, average_age) VALUES (?, ?, ?)", (unique_id, gender, average_age))
        conn.commit()
        save_button_placeholder.success(f"已記錄 ID: {unique_id} 的性別和平均年齡 {average_age:.2f}")

    cap.release()
