import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#ifrom PIL import Image

#st.set_page_config(page_title="Graph Edge Detector", layout="centered")

st.title("Graph Edge Detector")

uploaded_file = st.file_uploader("Upload a Graph Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as bytes and decode using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Could not decode the image. Please upload a valid image.")
        st.stop()
resized = cv2.resize(image, (900, 500))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = resized.copy()
edge_points = []

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        approx = cv2.approxPolyDP(cnt, epsilon=2, closed=False)
        for pt in approx:
            x, y = pt[0]
            edge_points.append((x, y))

edge_points = sorted(edge_points, key=lambda p: p[0])

for i in range(1, len(edge_points)):
    x1, y1 = edge_points[i - 1]
    x2, y2 = edge_points[i]
    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    dist_text = f"{abs(x2 - x1)}px,{abs(y2 - y1)}px"
    cv2.putText(output, dist_text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.imshow(output_rgb)
plt.axis('off')
plt.title("Annotated Graph Edges")
plt.show()
