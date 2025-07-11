import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="Graph Slope Detector", layout="centered")

st.title("Roadside Ditch Graph Slope Detector")
st.write("Upload a graph image. The app will highlight graph edges and calculate distances between them.")

uploaded_file = st.file_uploader("Upload Graph Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    height, width = image.shape[:2]
    
    st.image(image, caption="Uploaded Graph", use_column_width=True)

    # Grayscale and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sidebar: Scale input
    st.sidebar.subheader("Graph Scale Settings")
    x_min = st.sidebar.number_input("X-axis Min Value", value=0)
    x_max = st.sidebar.number_input("X-axis Max Value", value=45)
    y_min = st.sidebar.number_input("Y-axis Min Value", value=80)
    y_max = st.sidebar.number_input("Y-axis Max Value", value=85)

    x0_pixel = 80
    x45_pixel = width - 10
    pixels_per_meter_x = (x45_pixel - x0_pixel) / (x_max - x_min)

    y80_pixel = height - 80
    y85_pixel = 10
    pixels_per_meter_y = (y80_pixel - y85_pixel) / (y_max - y_min)

    edge_points = []

    # Filter and approximate edges
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            approx = cv2.approxPolyDP(cnt, epsilon=2, closed=False)
            for pt in approx:
                x, y = pt[0]
                edge_points.append((x, y))

    # Sort by X for slope traversal
    edge_points = sorted(edge_points, key=lambda p: p[0])

    # Calculate distances and annotate
    output_image = image.copy()
    distances = []

    for i in range(1, len(edge_points)):
        x1, y1 = edge_points[i - 1]
        x2, y2 = edge_points[i]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        dx_m = round(dx / pixels_per_meter_x, 2)
        dy_m = round(dy / pixels_per_meter_y, 2)

        distances.append({
            "Point #": i,
            "From (px)": f"{x1},{y1}",
            "To (px)": f"{x2},{y2}",
            "X Distance (px)": dx,
            "Y Distance (px)": dy,
            "X Distance (m)": dx_m,
            "Y Distance (m)": dy_m
        })

        # Draw line
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Annotate with distances
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(output_image, f"{dx_m}m, {dy_m}m", (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    st.image(output_image, caption="Edges with Distances", use_column_width=True)

    # Show data table
    if distances:
        df = pd.DataFrame(distances)
        st.subheader("Edge Distances Table")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="graph_edge_distances.csv",
            mime="text/csv"
        )
    else:
        st.warning("No edges detected in the image.")
else:
    st.info("Please upload a graph image to get started.")
