import streamlit as st
import cv2
import numpy as np
from PIL import Image # Pillow library for easier image handling with Streamlit

# --- Pothole Detection Logic (modified from previous example) ---
def process_image_for_potholes(img_array, blur_ksize=5, thresh_block_size=11, thresh_c=3, morph_ksize=5, min_area=150, max_area=5000):
    """
    Processes an image array to detect potential potholes using basic CV.

    Args:
        img_array (np.array): Input image as a NumPy array (from OpenCV).
        blur_ksize (int): Kernel size for Gaussian Blur (must be odd).
        thresh_block_size (int): Block size for adaptive thresholding (must be odd).
        thresh_c (int): Constant subtracted from the mean in adaptive thresholding.
        morph_ksize (int): Kernel size for morphological closing (must be odd).
        min_area (int): Minimum contour area to be considered a pothole.
        max_area (int): Maximum contour area to be considered a pothole.


    Returns:
        tuple: (processed_image_array, pothole_count)
               processed_image_array: Original image with bounding boxes drawn.
               pothole_count: Number of potential potholes detected.
    """
    if img_array is None:
        st.error("Error: Invalid image data received.")
        return None, 0

    output_img = img_array.copy() # Work on a copy

    # 1. Pre-processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 2. Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, thresh_block_size, thresh_c)

    # 3. Morphological Operations (Optional Cleaning)
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Find Contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter Contours & Draw Bounding Boxes
    pothole_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            pothole_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red rectangle
            cv2.putText(output_img, f'Pothole {pothole_count}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return output_img, pothole_count

# --- Streamlit App ---

st.set_page_config(page_title="Pothole Detection MVP", layout="wide")

st.title("🚧 Pothole Detection MVP App 🚧")
st.write("""
Upload an image of a road, and this simple app will try to detect potential potholes
using basic computer vision techniques (thresholding and contour detection).
*Note:* This is an MVP and highly sensitive to lighting, shadows, and road texture.
It may produce many false positives or miss actual potholes. Adjust parameters for better results.
""")

# --- Sidebar for Parameters ---
st.sidebar.header("⚙️ Detection Parameters (Tune Me!)")

# Ensure kernel sizes are odd
blur_k = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, step=2, help="Size of the blurring kernel (must be odd).")
thresh_block = st.sidebar.slider("Threshold Block Size", 3, 31, 11, step=2, help="Size of the neighborhood for adaptive thresholding (must be odd).")
thresh_const = st.sidebar.slider("Threshold Constant (C)", 1, 10, 3, help="Constant subtracted from the mean in thresholding.")
morph_k = st.sidebar.slider("Morphology Kernel Size", 3, 15, 5, step=2, help="Size of the kernel for morphological closing (must be odd).")
min_pothole_area = st.sidebar.slider("Min Pothole Area", 50, 1000, 150, help="Minimum pixel area to be considered a pothole.")
max_pothole_area = st.sidebar.slider("Max Pothole Area", 500, 10000, 5000, help="Maximum pixel area to be considered a pothole.")


# --- Main Area ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the image bytes into an OpenCV image (NumPy array)
    opencv_image = cv2.imdecode(file_bytes, 1) # 1 means load in color

    if opencv_image is None:
        st.error("Could not decode image. Please upload a valid JPG, PNG, or JPEG file.")
    else:
        st.write("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            # Display the original image. Convert BGR (OpenCV default) to RGB for Streamlit.
            st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption='Uploaded Image.', use_column_width=True)

        with col2:
            st.subheader("Detection Result")
            with st.spinner('Processing image...'):
                # Process the image using the selected parameters
                processed_image, count = process_image_for_potholes(
                    opencv_image,
                    blur_ksize=blur_k,
                    thresh_block_size=thresh_block,
                    thresh_c=thresh_const,
                    morph_ksize=morph_k,
                    min_area=min_pothole_area,
                    max_area=max_pothole_area
                )

            if processed_image is not None:
                 # Display the processed image. Convert BGR to RGB.
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption='Processed Image with Detections.', use_column_width=True)
                if count > 0:
                    st.success(f"Detected {count} potential pothole(s) based on current parameters.")
                else:
                    st.info("No potential potholes detected with current parameters.")
            else:
                st.error("Image processing failed.")
else:
    st.info("Please upload an image file to begin detection.")

st.write("---")
st.markdown("ONE POTHOLE AT A TIME!!.")
