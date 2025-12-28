import os
import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime

print("hi--------------")

# ====== AUTO CREATE FOLDERS ======
os.makedirs("known_faces", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

# ====== LOAD KNOWN FACES FROM EXCEL ======
excel_file = r"C:\Users\dhara\frp\faces_names.xlsx"
if not os.path.exists(excel_file):
    pd.DataFrame(columns=["name", "image"]).to_excel(excel_file, index=False)
    print(f"[INFO] Created empty {excel_file}. Please add known faces and rerun.")
    exit()

df = pd.read_excel(excel_file)
known_faces = [{"name": row["name"], "image": row["image"]} for _, row in df.iterrows()]

# ====== START VIDEO CAPTURE ======
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="known_faces",
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=True,
            silent=True
        )

        if not result[0].empty:
            name = os.path.basename(result[0].iloc[0]['identity']).split(".")[0]
            cv2.putText(frame, f"Known: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Save new unknown face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"unknown_faces/person_{timestamp}.jpg"
            cv2.imwrite(new_filename, frame)
            cv2.putText(frame, "Unknown - Saved", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        print("[ERROR]", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
    
     