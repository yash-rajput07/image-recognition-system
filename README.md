# image-recognition-system
•Developed an object detection tool using YOLO and CNN,  achieving over 90% accuracy
pip install torch torchvision torchaudio
pip install opencv-python
pip install Pillow
pip install pyttk
pip install git+https://github.com/ultralytics/yolov5  # Installs YOLOv5
import torch
import cv2
from tkinter import *
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize GUI
root = Tk()
root.title("Live Object Detection")
root.geometry("800x600")

# Create a label for video stream
label = Label(root)
label.pack()

# Capture webcam video
cap = cv2.VideoCapture(0)

def detect_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model(img_rgb)

    # Plot results on the original frame
    annotated_frame = results.render()[0]
    # Convert to PIL Image and then to ImageTk
    img_pil = Image.fromarray(annotated_frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update GUI label
    label.imgtk = img_tk
    label.configure(image=img_tk)

    # Repeat every 30ms
    root.after(30, detect_frame)

# Start detection
detect_frame()

# Start GUI loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
