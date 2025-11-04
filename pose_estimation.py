import cv2
import time
import mediapipe as mp
from picamera2 import Picamera2
from utils import (
    get_pose_detector,
    draw_pose_landmarks_on_image,
)

INTERVAL = 0.1      # Capture a frame every <INTERVAL> seconds
MARKER_SIZE = 5     # size of the landmark markers to draw

def get_picam2_instance():
    """ Initialize and return a Picamera2 instance configured for square preview. """
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={
        "format": "RGB888", 
        "size": (200, 200),
    })
    picam2.configure(camera_config)
    
    return picam2

def main():

    picam2 = get_picam2_instance()
    picam2.start()
    pose_detector = get_pose_detector()

    window_name = "Pi Camera Stream "
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_time = 0

    try:
        while True:
            now = time.time()
            
            if now - last_time >= INTERVAL:
                last_time = now
                # make sure it is squre image
                ori_square_frame = picam2.capture_array() # returns a numpy array in RGB order
                mp_pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ori_square_frame)

                ### Pose detection
                detection_result = pose_detector.detect(mp_pose_image)
                if not detection_result.pose_landmarks:
                    print("No pose landmarks detected in the image.")
                    annotated_image = ori_square_frame
                else:
                    annotated_image = draw_pose_landmarks_on_image(
                        original_image=ori_square_frame,
                        detection_result=detection_result,
                        marker_size=MARKER_SIZE,
                    )

                cv2.imshow(window_name, annotated_image)

            # check for key press
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF:
                # press any key to exit
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
