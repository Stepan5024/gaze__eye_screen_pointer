import cv2
from batch_face.face_detection import RetinaFace
from batch_face import drawLandmark_multiple, LandmarkPredictor
import os

def generate_unique_filename(directory, base_name, extension):
    """Generate a unique filename in the specified directory."""
    i = 0
    while True:
        new_filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        i += 1

if __name__ == "__main__":
    file_name = "stepa_red_short"
    predictor = LandmarkPredictor(0)
    detector = RetinaFace(0)

    imgname = f"examples/{file_name}.jpg"
    img = cv2.imread(imgname)

    faces = detector(img, cv=True)

    if len(faces) == 0:
        print("NO face is detected!")
        exit(-1)

    results = predictor(faces, img, from_fd=True)

    for face, landmarks in zip(faces, results):
        img = drawLandmark_multiple(img, face[0], landmarks)

    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.waitKey(27)
    cv2.waitKey(ord('q'))
     # Save the processed image
    save_directory = "examples/predicted"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)  # Create the directory if it doesn't exist

    unique_filename = generate_unique_filename(save_directory, file_name, "jpg")
    cv2.imwrite(os.path.join(save_directory, unique_filename), img)
    print(f"Image saved as {unique_filename}")
