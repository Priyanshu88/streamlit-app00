import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

@st.cache(persist=True)
class Streamlit_YOLOV7(SingleInference_YOLOV7):

    def __init__(self,):
        self.logging_main = logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self, img_size, path_yolov7_weights, device_i='cpu'):
        super().__init__(img_size, path_yolov7_weights, device_i=device_i)

    def main(self):
        st.title('Custom YoloV7 Object Detector')
        st.subheader("""Upload an image and run YoloV7 on it. This Model was Trained to Detect the Following Classes from Digital Architectural Floor Plans.\n""")
        text_i_list = []
        for i, name_i in enumerate(self.names):
            text_i_list.append(f'{i}: {name_i}\n')
        st.selectbox('Classes', tuple(text_i_list))
        self.conf_selection = st.selectbox('Confidence Threshold', tuple([0.1, 0.25, 0.5, 0.75, 0.95]))
        st.markdown('YoloV7 on streamlit. Demo of object detection with YoloV7 with a web application.')
        self.load_image_st()
        predictions = st.button('Predict on the image?')
        if predictions:
            self.predict()
            predictions = False

    def load_image_st(self):
        uploaded_img = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
        if type(uploaded_img) != type(None):
            self.img_data = uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0 = Image.open(BytesIO(self.img_data))  # .convert('RGB')
            self.im0 = np.array(self.im0)
            return self.im0
        elif type(self.im0) != type(None):
            return self.im0
        else:
            return None

    def predict(self):
        self.conf_thres = self.conf_selection
        st.write('Loading image')
        self.load_cv2mat(self.im0)
        st.write('Making inference')
        self.inference()
        self.img_screen = Image.fromarray(self.image).convert('RGB')
        self.capt = 'DETECTED:'
        if len(self.predicted_bboxes_PascalVOC) > 0:
            for item in self.predicted_bboxes_PascalVOC:
                name = str(item[0])
                name_count = str((item[0]).count(item[0]))
                conf = str(round(100*item[-1], 2))
                # self.capt = self.capt + ' Name = '+name+name_count+' Confidence = '+conf+'%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        self.image = None
        unique_labels=set([item[0] for item in self.predicted_bboxes_PascalVOC])
        for label in unique_labels:
            count=sum([1 for item in self.predicted_bboxes_PascalVOC if item[0]==label])
            st.write(f'Number of {label} in the image is {count}')

if __name__ == '__main__':
    app = Streamlit_YOLOV7()
    img_size = 1056
    path_yolov7_weights = "weights/best.pt"
    app.capt = "Initial Image"
    app.new_yolo_model(img_size, path_yolov7_weights)
    app.conf_thres = 0.65
    app.load_model()  # Load the yolov7 model
    app.main()
