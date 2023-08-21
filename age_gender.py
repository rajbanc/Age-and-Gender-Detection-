import streamlit as st
import cv2
import numpy as np

def highlightFace(net, frame, conf_threshold=0.8):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_age_gender(face, genderNet, ageNet, genderList, ageList):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    return gender, age

def put_text_on_image(resultImg, gender, age):
    cv2.putText(resultImg, f'Gender: {gender}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resultImg, f'Age: {age[1:-1]} years', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__": 

    st.title("Age and Gender Detection with OpenCV and Streamlit")

    # Load pre-trained models
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"

    ageList = ['(0-2)', '(3-6)', '(7-10)','(11-20)', '(21-30)', '(31-40)', '(41-50)', '(51-60)','(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    camera_placeholder = st.empty()

    padding = 20
    # Create a file uploader to allow users to upload an image or use the camera
    option = st.sidebar.radio("Choose an option:", ("Upload an image", "Use camera"))

    if option == "Upload an image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            # Read the uploaded image
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            # Display the uploaded image
            camera_placeholder.image(image, channels="BGR")
            # Process the image using your highlightFace function
            resultImg, faceBoxes = highlightFace(faceNet, image)
            if not faceBoxes:
                st.warning("No face detected")

            for faceBox in faceBoxes:
                face = image[max(0, faceBox[1] - padding):
                             min(faceBox[3] + padding, image.shape[0] - 1), max(0, faceBox[0] - padding)
                             :min(faceBox[2] + padding, image.shape[1] - 1)]

                gender, age = detect_age_gender(face, genderNet, ageNet, genderList, ageList)

                cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), int(round(image.shape[0] / 150)), )
                cv2.putText(resultImg, f'Gender: {gender}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(resultImg, f'Age: {age[1:-1]} years', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            camera_placeholder.image(resultImg, channels="BGR", caption="Detecting age and gender")
    else:  # Use camera
        # Capture video from the default camera (you can specify a different camera if needed)
        video = cv2.VideoCapture(0)

        while True:
            hasFrame, frame = video.read()

            if not hasFrame:
                st.warning("Camera not found!")
                break

            # Process the frame using your highlightFace function
            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if not faceBoxes:
                continue
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding):
                              min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                              :min(faceBox[2] + padding, frame.shape[1] - 1)]

                gender, age = detect_age_gender(face, genderNet, ageNet, genderList, ageList)

                put_text_on_image(resultImg, gender, age)

            camera_placeholder.image(resultImg, channels="BGR", caption="Detecting age and gender")

        video.release()
        cv2.destroyAllWindows()
