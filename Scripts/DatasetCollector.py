import cv2
import os
save_path = r"D:\SignLanguageTranslator\Data\Train"
label = input("Enter the label for the sign (e.g., 'Hello'): ")
label_path = os.path.join(save_path, label)
os.makedirs(label_path, exist_ok=True)
cap = cv2.VideoCapture(0)  # Open webcam
print("Press 's' to start capturing images. Press 'q' to quit.")
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):  # Start saving images
        img_name = os.path.join(label_path, f"{label}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
    elif key == ord('q'):  # Quit
        break
cap.release()
cv2.destroyAllWindows()

