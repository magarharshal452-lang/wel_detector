import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
# Load YOLO
yolo_model = YOLO("best.pt")

# Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,2,1), torch.nn.ReLU(),
            torch.nn.Conv2d(16,32,3,2,1), torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,2,1), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64,32,3,2,1,1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32,16,3,2,1,1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,3,3,2,1,1), torch.nn.Sigmoid()
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
autoencoder.eval()

st.title("Weld Defect Detection")

uploaded_file = st.file_uploader("Upload Weld Image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image")

    results = yolo_model.predict(image, imgsz=320)
    boxes = results[0].boxes

    if len(boxes) > 0:
        labels = results[0].names
        detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]

        if "Defect" in detected_classes or "Bad Weld" in detected_classes:
            st.write("Result: DEFECTIVE WELD")
        else:
            st.write("Result: GOOD WELD")

        st.image(results[0].plot(), caption="Detection Result")

    else:
        img = image.resize((128,128))
        img = np.array(img)/255.0
        img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

        recon = autoencoder(img)
        loss = torch.mean((img - recon)**2).item()

        if loss > 0.006:
            st.write("Result: DEFECTIVE (Anomaly)")
        else:
            st.write("Result: GOOD WELD")

        st.write(f"Anomaly Score: {loss:.6f}")
