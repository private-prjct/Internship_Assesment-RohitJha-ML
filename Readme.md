# 🧠 Object Detection Streamlit Web App

*A beginner-friendly yet powerful YOLOv8-based web interface for object detection on images and videos.*

---

## 📘 Overview

This project provides an **interactive Streamlit-based web application** that detects objects in both **images** and **videos** using a **YOLOv8 model**.
It allows users to upload media, view annotated outputs, and analyze object counts or potential rule violations (like vehicles crossing a stop line).

The app is modular, easy to extend, and structured cleanly for clarity and scalability.

---

## 🧩 Project Structure

```
Object-Detection/
│
├── App/
│   ├── constants.py       # Stores fixed configuration values (e.g., confidence threshold, output paths)
│   ├── detection.py       # Core logic for processing images and videos
│   ├── utils.py           # Helper utilities (saving videos, formatting outputs, etc.)
│
├── Model/
│   └── yolo.py            # Loads YOLOv8 model and manages inference
|       
|Output/
|   └── annotated_images   # Saves the annotated images
|   ├── annotated_videos   # Saves the annotated videos
│   
├── main.py                # Streamlit web interface for user interaction
│
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/rohitjhaofficial1503-wq/Object-Detection.git
   cd Object-Detection
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run main.py
   ```

---

## 🚀 How It Works (Methodology)

The workflow is simple yet efficient:

1. **User uploads** an image or video via the Streamlit interface (`main.py`).
2. The uploaded file is sent to the **processing module** (`App/detection.py`).
3. The **YOLOv8 model** (from `Model/yolo.py`) performs detection on each frame or image.
4. The detections are then:

   * Visualized (annotated with bounding boxes and class labels)
   * Counted (how many instances of each class were found)
   * Saved in a structured output folder
5. The **annotated media** (image/video) is displayed back on the Streamlit interface.

---

## 🧠 Core Logic Explained

### ### 1️⃣ `process_image(uploaded_file, conf=CONFIDENCE_THRESHOLD)`

**Purpose:**
Processes a single uploaded image and returns:

* Annotated image with bounding boxes
* Count of each detected object
* Raw detection boxes for further logic if needed

**Steps:**

1. Opens the uploaded image with **PIL (Python Imaging Library)**.

2. Runs YOLO detection:

   ```python
   detection_results = model(image, conf=conf)
   ```

   * `model` → YOLOv8 model object imported from `Model/yolo.py`
   * `conf` → Minimum confidence threshold (from `App/constants.py`)

3. YOLO returns a list of results (each containing bounding boxes, class IDs, confidence scores).

4. The model draws bounding boxes using:

   ```python
   annotated_image = detection_results[0].plot()
   ```

5. Counts are computed:

   ```python
   for cls in detection_results[0].boxes.cls:
       class_name = model.names[int(cls)]
       counts[class_name] = counts.get(class_name, 0) + 1
   ```

   This generates a dictionary like:

   ```python
   {'car': 3, 'person': 2, 'dog': 1}
   ```

6. The image is color-corrected (BGR → RGB) and saved in `OUTPUT_IMAGE_FOLDER`.

7. Returns the annotated frame, class counts, and YOLO bounding boxes.

**Main Variables:**

| Variable            | Purpose                             |
| ------------------- | ----------------------------------- |
| `uploaded_file`     | Image uploaded by the user          |
| `conf`              | Confidence threshold for detection  |
| `image`             | PIL Image object loaded from upload |
| `detection_results` | YOLO model predictions              |
| `annotated_image`   | Image array with bounding boxes     |
| `counts`            | Dictionary of object counts         |
| `output_path`       | Save path for annotated image       |

---

### 2️⃣ `process_video(uploaded_file, conf=CONFIDENCE_THRESHOLD)`

**Purpose:**
Processes an uploaded video frame by frame, performs YOLO detection, and outputs an annotated video file.

**Steps:**

1. Opens the video with OpenCV:

   ```python
   cap = cv2.VideoCapture(uploaded_file)
   ```

2. Initializes:

   ```python
   listof_annotated_frames = []
   violation_count = 0
   ```

   * `listof_annotated_frames` → stores processed frames
   * `violation_count` → placeholder for stop-line rule detection logic

3. Reads frames in a loop:

   ```python
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
   ```

4. YOLOv8 runs inference on each frame:

   ```python
   results = model(frame, conf=conf, stream=True)
   ```

5. Each result is visualized and appended:

   ```python
   for r in results:
       annotated_frame = r.plot()
       listof_annotated_frames.append(annotated_frame)
   ```

6. A timestamped output filename is created:

   ```python
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_filename = f"detected_{timestamp}.mp4"
   ```

7. The final video is written using the helper:

   ```python
   video_output_path = save_video(listof_annotated_frames, output_path)
   ```

8. The function releases resources and returns the path to the annotated video.

**Main Variables:**

| Variable                  | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| `uploaded_file`           | Path to uploaded video                           |
| `cap`                     | OpenCV video capture object                      |
| `frame`                   | Current frame from video                         |
| `results`                 | YOLOv8 detection results for frame               |
| `annotated_frame`         | Frame after drawing boxes                        |
| `listof_annotated_frames` | Stores all annotated frames                      |
| `timestamp`               | Used for naming output video uniquely            |
| `output_path`             | Full path to save output video                   |
| `violation_count`         | Placeholder for future stop-line violation logic |

---

## 📊 Future Enhancements

* [ ] Add **stop-line detection** logic to count vehicles crossing restricted boundaries
* [ ] Integrate **object tracking (DeepSORT)** for persistent detection IDs
* [ ] Include **real-time webcam inference**
* [ ] Display **detection statistics** on dashboard (Streamlit widgets)

---

## 🧰 Tech Stack

| Layer             | Tools Used             |
| ----------------- | ---------------------- |
| **Frontend**      | Streamlit, HTML/CSS    |
| **Backend**       | Python, OpenCV, PIL    |
| **Model**         | YOLOv8 (Ultralytics)   |
| **Data Handling** | NumPy, Pandas          |
| **Utilities**     | datetime, os, tempfile |

---

## 🏁 Output Example

* **Input:** Image/Video uploaded by user
* **Output:** Annotated image/video file with detected objects and counts displayed in UI

---

## ✍️ Author

**Rohit Jha**
*Developer & ML Enthusiast*

> A hands-on beginner project built to understand computer vision workflows with YOLOv8, OpenCV, and Streamlit.

---

### ⭐ Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Streamlit](https://streamlit.io)
* [OpenCV](https://opencv.org)

---

**Ready to detect objects in seconds. 🚀**
