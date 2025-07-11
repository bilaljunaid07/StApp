import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Graph Slope Detector", layout="centered")

st.title("Roadside Ditch Graph Slope Detector")
st.write("Upload your graph image and get slope distances in meters.")

uploaded_file = st.file_uploader("Upload Graph Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show original image
    st.image(image, caption="Uploaded Graph", use_column_width=True)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slope_points = []
    for cnt in contours:
        for point in cnt:
            x, y = point[0]
            slope_points.append((x, y))

    slope_points = sorted(slope_points, key=lambda p: p[0])

    # User input for graph scales
    st.sidebar.subheader("Graph Scale Settings")
    x_min = st.sidebar.number_input("X-axis Min Value", value=0)
    x_max = st.sidebar.number_input("X-axis Max Value", value=45)
    y_min = st.sidebar.number_input("Y-axis Min Value", value=80)
    y_max = st.sidebar.number_input("Y-axis Max Value", value=85)

    height, width = image.shape[:2]
    x0_pixel = 80
    x45_pixel = width - 10
    pixels_per_meter_x = (x45_pixel - x0_pixel) / (x_max - x_min)

    y80_pixel = height - 80
    y85_pixel = 10
    pixels_per_meter_y = (y80_pixel - y85_pixel) / (y_max - y_min)

    # Slope distances
    slopes_data = []
    for i in range(1, len(slope_points)):
        x1, y1 = slope_points[i - 1]
        x2, y2 = slope_points[i]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > 5 or dy > 5:
            slopes_data.append({
                "Slope No.": len(slopes_data) + 1,
                "X Distance (px)": dx,
                "Y Difference (px)": dy,
                "X Distance (m)": round(dx / pixels_per_meter_x, 2),
                "Y Difference (m)": round(dy / pixels_per_meter_y, 2)
            })

    if slopes_data:
        df = pd.DataFrame(slopes_data)
        st.subheader("Calculated Slopes")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name='slope_distances_report.csv',
            mime='text/csv'
        )
    else:
        st.warning("No slopes detected in the uploaded image.")
else:
    st.info("Please upload a graph image to get started.")
