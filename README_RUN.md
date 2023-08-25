**INDIVIDUAL RESERACH THESIS: TRACKING OBJECT BEHAVIOUR BY LEVERAGING A SINGLE CAMERA SENSOR PLACED ON THE ROAD INFRASTRUCTURE**

NAME: CHANDNI SAHA

STUDENT ID: s387796

MSc in COMPUTATIONAL AND SOFTWARE TECHNIQUES IN ENGINEERING (COMPUTATIONAL INTELLIGENCE IN DATA ANALYTICS)
!git clone https://github.com/Chandni12300/object_detection_and_tracking_deepsort_yolov8.git
%pip install ultralytics
import ultralytics
ultralytics.checks()
%cd /content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect
***DOWNLOADING COCO DATASET***
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="VJDeMwjnn3DSw8bIdkPJ")
project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
dataset = project.version(2).download("yolov5")

***TRAINING THE MODEL WITH 50 EPOCHS***
!yolo train model=yolov8l.pt data=/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/vehicles-2/data.yaml epochs=50 imgsz=640
!pwd
%cd /content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2
!ls
***DISPLAYING THE RESULTS OBTAINED AFTER MODEL TRAINING***
from IPython.display import Image
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/F1_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/PR_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/P_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/R_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/confusion_matrix.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/confusion_matrix_normalized.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/labels_correlogram.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/results.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch0.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch1.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch2.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch6600.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch6601.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/train_batch6602.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch0_labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch0_pred.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch1_labels.jpg')

Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch1_pred.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch2_labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/runs/detect/train2/val_batch2_pred.jpg')
%cd /content/object_detection_and_tracking_deepsort_yolov8
!pip install -e '.[dev]'
%cd /content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect
!gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
!unzip 'deep_sort_pytorch.zip'
HOME = '/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect'
***CONDUCTING OBJECT DETECTION AND TRAJECTORY TRACKING***
!python predict.py model=yolov8l.pt source='test1.mp4'
!rm "/content/result_compressed.mp4"
***EXECUTION AFTER CONDUCTING OBJECT DETECTION AND TRAJECTORY TRACKING***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train/test1.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
***OBJECT DETECTION VALIDATION***
!python objectdetectionvalidation.py model=yolov8l.pt
***RESULT OBTAINED AFTER OBJECT DETECTION VALIDATION***
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/F1_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/PR_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/P_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/R_curve.png')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/val_batch0_labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/val_batch0_pred.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/val_batch1_labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/val_batch2_labels.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train2/val_batch2_pred.jpg')
!pip install filterpy
***PREDICTION OF FUTURE TRAJECTORY USING KALMAN FILTER***
!python futuretrajectoryprediction.py model=yolov8l.pt source='test1.mp4'
!rm "/content/result_compressed.mp4"
***EXECUTION AFTER FUTURE PREDICTION***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train3/test1.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
***COLLISON RISK ANALYSIS***
!python riskanalysis.py model=yolov8l.pt source='test1.mp4'
!rm "/content/result_compressed.mp4"
***EXECUTION OF COLLISION RISK ANALYSIS***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train5/test1.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
***COLLISION RISK VALIDATION***
!python collisionvalidation.py model=yolov8l.pt
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train6/adit_mp4-105_jpg.rf.19f2d6c147c5bba0097ecedd310449be.jpg')
***RESULT OBTAINED AFTER VALIDATION***
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train6/adit_mp4-1047_jpg.rf.754fd8f5a99e6ce063f875f9d67c15dd.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train6/adit_mp4-1400_jpg.rf.ced9d503af6b0772186c98739ae83f8b.jpg')
Image(filename = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train6/adit_mp4-1795_jpg.rf.2f10db200dbecb9e5741b4696780a3f6.jpg')
%cd {HOME}
***REAL-TIME VALIDATION USE CASES FOR RISK ANALYSIS***
!python riskanalysis.py model=yolov8l.pt source='/content/risk_analysis_testcases_Input_Video.mp4'
!rm "/content/result_compressed.mp4"
***EXECUTION OF REAL-TIME USECASES***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train11/risk_analysis_testcases_Input_Video.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
!pip install filterpy
***PREDICTION OF FUTURE COLLISION RISK ANALYSIS***
!python predictionriskanalysis.py model=yolov8l.pt source='/content/risk_analysis_testcases_Input_Video.mp4'
***TRAFFIC MANAGEMENT***
***COUNT OF VEHICLES***
!python countofvehicles.py model=yolov8l.pt source='test1.mp4'
!rm "/content/result_compressed.mp4"
***EXECUTION OF COUNT OF VEHICLES***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train13/test1.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
***COUNT OF VEHICLES VISUALISATION***
!python tmanagementvisualisation.py model=yolov8l.pt source='test1.mp4'
***DISPLAYING GRAPH***

from IPython.display import display, HTML
# Loading and displaying the HTML file
with open('/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/Graphs/entering_graph.html', 'r') as file:
    html_content = file.read()
display(HTML(html_content))

with open('/content/object_detection_and_tracking_deepsort_yolov8/ultralytics/yolo/v8/detect/Graphs/leaving_graph.html', 'r') as file:
    html_content = file.read()
display(HTML(html_content))
***CONGESTION***
!python congestion.py  source='test1.mp4'
!rm "/content/result_compressed.mp4"
***CONGESTION EXECUTION***
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/object_detection_and_tracking_deepsort_yolov8/runs/detect/train15/test1.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
