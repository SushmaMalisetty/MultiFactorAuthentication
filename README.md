# 🛡️ Multi-Factor Biometric Authentication System  
### (Face Recognition + Hand Gesture Verification)

---

## 📌 Project Overview
This mini project implements a **Multi-Factor Biometric Authentication System** using **Face Recognition** as the first factor and **Hand Gesture Verification** as the second factor.  
The system improves security by combining two biometric methods instead of traditional password-based authentication.

The project is **laptop-friendly**, uses a **webcam**, and demonstrates a **real-world security concept** using Computer Vision.

---

## 🎯 Objective
- To build a **password-less authentication system**
- To verify user identity using **biometric features**
- To demonstrate **multi-factor authentication**
- To understand real-time computer vision security systems

---

## 🔐 Authentication Factors Used

### 1️⃣ Face Authentication
- Verifies the user using stored facial images
- Uses **LBPH (Local Binary Pattern Histogram)** algorithm
- Requires the user to hold the face steady for verification

### 2️⃣ Gesture Authentication
- Verifies the user using a **secret hand gesture (open palm)**
- Implemented using OpenCV-based image processing
- Gesture must be held steadily to avoid false detection

Access is granted **only if both authentication factors succeed**.

---

## 🧠 System Workflow
1. User registration (face dataset creation)
2. Face authentication using live camera
3. Gesture authentication using live camera
4. Access granted or denied

---

## 🗂️ Project Structure
```text
MultiFactorAuthentication/
│
├── dataset/
│   └── user1/                  # Face dataset images
│
├── face_dataset.py             # Face registration (dataset creation)
├── delete_dataset.py           # Deletes existing face dataset
├── face_auth.py                # Face authentication (standalone)
├── gesture_auth.py             # Gesture authentication (standalone)
├── main.py                     # Complete multi-factor authentication system
│
├── README.md
└── requirements.txt

## 🛠️ Technologies Used
- **Python 3.10**
- **OpenCV (opencv-contrib-python)**
- **NumPy**
- **Computer Vision concepts**
- **Laptop Webcam**

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv310
venv310\Scripts\activate

### 2️⃣ Install Required Libraries
```bash
pip install opencv-contrib-python numpy

## ▶️ How to Run the Project

### 🔹 Step 1: Register Face (Face Dataset Creation)
```bash
python face_dataset.py

### 🔹 Gesture Authentication (Standalone)
```bash
python gesture_auth.py

#### 🔹 Complete Multi-Factor Authentication System
```bash
python main.py
