import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def cvt_raw_image_squre(
    raw_image: np.ndarray,
) -> np.ndarray:
    """ Convert the raw image to a square image by cropping it to the center. """
    h, w, _ = raw_image.shape
    if h != w:
        half_size = min(h, w) // 2
        center_h, center_w = h // 2, w // 2
        square_image = raw_image[
            center_h - half_size:center_h + half_size, 
            center_w - half_size:center_w + half_size,
        ].astype(np.uint8)
    else:
        square_image = raw_image.astype(np.uint8)
    square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
    return square_image


###### pose detection ######


def get_pose_detector():
    base_options = python.BaseOptions(
        model_asset_path='checkpoints/pose_landmarker_lite.task',
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        num_poses=1, # the detector only detects one person at a time
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def get_important_joints(
    img_height: int,
    img_width: int,
    detected_person,
) -> Dict[str, np.int32]:
    """ Extracts the coordinates of important joints.

    :param img_height & img_height: the height and width of original image
    :param detected_person: A single detected person from the detection result, which contains pose landmarks.
    
    :output important_joints_coords: contains the coordinates of important joints
    """
    pose_landmark_description = dict(
        # arm landmarks
        left_shoulder=11,
        right_shoulder=12,
        left_elbow=13,
        right_elbow=14,
        left_wrist=15,
        right_wrist=16,
        # hand landmarks
        left_pinky=17,
        right_pinky=18,
        left_index=19,
        right_index=20,
        left_thumb=21,
        right_thumb=22,
    )
    important_joints_coords = dict()
    for joint_name, idx in pose_landmark_description.items():
        joint = detected_person[idx]
        joint_coords = np.array([int(joint.x * img_width), int(joint.y * img_height)]).astype(np.int32)
        important_joints_coords[joint_name] = joint_coords
    return important_joints_coords


def get_end_effector_coords(
    important_joints_coords: Dict[str, np.int32],
    lr: str,  # 'left' or 'right'
) -> np.ndarray:
    """ Get the coordinates of the end effectors (i.e. wrist) for the specified hand. """
    wrist = important_joints_coords[f'{lr}_wrist']
    shoulder = important_joints_coords[f'{lr}_shoulder']
    return wrist - shoulder


def get_hand_center(
    pinky: np.int32,
    index: np.int32,
    thumb: np.int32,
) -> np.int32:
    """ Calculate the center of the hand based on the positions of pinky, index, and thumb landmarks. """
    return ((pinky + index + thumb) / 3).astype(np.int32)


def get_hand_rotation_angle(
    wrist: np.int32,
    elbow: np.int32,
) -> float:
    """ Calculate the angle of the hand based on wrist and elbow positions. """
    diff_vec = wrist - elbow  # vector from elbow to wrist
    angle_rad = np.arctan2(diff_vec[1], diff_vec[0]) # dy, dx
    angle_deg = np.degrees(angle_rad)
    rot_deg = 90 + angle_deg # adjust to make the hand upright
    return float(rot_deg)


###### gesture recognition ######


def get_gesture_recognizer():
    base_options = python.BaseOptions(model_asset_path='src/pose_gesture_perception/pose_gesture_perception/checkpoints/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,  # the recognizer only recognizes one hand at a time
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    return recognizer


def enlarge_rotate_crop_hand(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    edge_length: int,
    angle_deg: float, 
    output_size: int = 224,
) -> np.ndarray:
    """ Crop an enlarged square region around the hand, rotate it to align the hand vertically,
    and center-crop or resize the result to a square image for gesture recognition.

    :param image: The input image (BGR or RGB as needed).
    :param cx, cy, size: center x, center y, and side length (in pixels) of the hand region.
    :param angle_deg: Rotation angle in degrees. Positive values mean counter-clockwise rotation.
    :param output_size: Output square size (e.g., 224 for 224x224).

    :output final_crop: The rotated and cropped hand image (output_size x output_size).
    """
    enlarged_size = int(edge_length * np.sqrt(2))  # Enlarge by sqrt(2) to fit the hand after rotation

    # Calculate coordinates for the enlarged crop
    x1 = int(center_x - enlarged_size // 2)
    y1 = int(center_y - enlarged_size // 2)
    x2 = x1 + enlarged_size
    y2 = y1 + enlarged_size

    # Create a blank patch and copy the image crop into it (handle borders)
    hand_patch = np.zeros((enlarged_size, enlarged_size, 3), dtype=image.dtype)
    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    src_x2 = min(x2, image.shape[1])
    src_y2 = min(y2, image.shape[0])
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    hand_patch[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # Rotate the patch
    rot_mat = cv2.getRotationMatrix2D((enlarged_size // 2, enlarged_size // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(hand_patch, rot_mat, (enlarged_size, enlarged_size), borderValue=(0,0,0))

    # Center-crop to original size
    start = enlarged_size // 2 - edge_length // 2
    end = start + edge_length
    final_crop = rotated[start:end, start:end]

    # Resize to model input size
    final_crop = cv2.resize(final_crop, (output_size, output_size))
    return final_crop


def get_hand_image(
    ori_square_img: np.ndarray,
    important_joints_coords: Dict[str, np.int32],
    limit_edge: int,
    lr: str,# 'left' or 'right'
):
    """ Extract the right hand image from the original square image. """
    hand_center = get_hand_center(
        pinky=important_joints_coords[f'{lr}_pinky'],
        index=important_joints_coords[f'{lr}_index'],
        thumb=important_joints_coords[f'{lr}_thumb'],
    )
    if hand_center[0] >= limit_edge or hand_center[1] >= limit_edge:
        print(f"Hand center {hand_center} is out of bounds for the image size {limit_edge}.")
        return None
    
    hand_rot_angle = get_hand_rotation_angle(
        wrist=important_joints_coords[f'{lr}_wrist'],
        elbow=important_joints_coords[f'{lr}_elbow'],
    )
    hand_image = enlarge_rotate_crop_hand(
        image=ori_square_img,
        center_x=hand_center[0],
        center_y=hand_center[1],
        edge_length=300,
        angle_deg=hand_rot_angle, 
        output_size=224,
    )
    return hand_image


def get_gesture(
    recognition_result,
) -> str:
    """ Get result of detected gesture class."""

    if len(recognition_result.gestures) == 0:
        print("No gesture detected in the image.")
        return None
    gesture = recognition_result.gestures[0][0].category_name
    return gesture


###### visualization functions ######


def draw_pose_landmarks_on_image(
    original_image: np.ndarray, 
    detection_result,
    marker_size: int = 5,
):
    """ ONly draw important pose landmarks on the given RGB image. """

    important_joints_coords = get_important_joints(
        img_height=original_image.shape[0],
        img_width=original_image.shape[1],
        detected_person=detection_result.pose_landmarks[0],
    )
    annotated_image = np.copy(original_image)
    # arms
    right_arm_joints=['right_shoulder', 'right_wrist', 'right_elbow']
    left_arm_joints=['left_shoulder', 'left_wrist', 'left_elbow']
    for joint_name, joint_coords in important_joints_coords.items():
        if joint_name in right_arm_joints:
            cv2.circle(
                img=annotated_image,
                center=joint_coords,
                radius=marker_size,
                color=(0, 255, 255), 
                thickness=-1,
            )
        elif joint_name in left_arm_joints:
            cv2.circle(
                img=annotated_image,
                center=joint_coords,
                radius=marker_size,
                color=(255, 0, 255), 
                thickness=-1,
            )
    # hand center
    right_hand_center = get_hand_center(
        pinky=important_joints_coords['right_pinky'],
        index=important_joints_coords['right_index'],
        thumb=important_joints_coords['right_thumb'],
    )
    cv2.circle(
        img=annotated_image,
        center=right_hand_center,
        radius=marker_size,
        color=(0, 120, 120), 
        thickness=-1,
    )
    left_hand_center = get_hand_center(
        pinky=important_joints_coords['left_pinky'],
        index=important_joints_coords['left_index'],
        thumb=important_joints_coords['left_thumb'],
    )
    cv2.circle(
        img=annotated_image,
        center=left_hand_center,
        radius=marker_size,
        color=(120, 0, 120), 
        thickness=-1,
    )
    return annotated_image


def draw_gesture_landmarks_on_image(
    rgb_image: np.ndarray, 
    recognition_result,
) -> np.ndarray:
    """ Draws pose landmarks on the given RGB image."""

    if len(recognition_result.gestures) == 0:
        print("No gesture detected in the image.")
        return rgb_image
    
    gesture = recognition_result.gestures[0][0].category_name
    print(f"Detected gesture: {gesture}")

    detected_hand = recognition_result.hand_landmarks[0]
    annotated_image = np.copy(rgb_image)

    for landmark in detected_hand:
        joint_coords = (int(landmark.x * rgb_image.shape[1]), int(landmark.y * rgb_image.shape[0]))
        # Draw the landmark as a circle on the annotated image.
        cv2.circle(
            img=annotated_image,
            center=joint_coords,
            radius=5, 
            color=(255, 255, 0), 
            thickness=-1,
        )
    plt.imshow(annotated_image)


def main():

    ### Create webcam capture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ### Create detectors
    pose_detector = get_pose_detector()
    gesture_recognizer = get_gesture_recognizer()

    while True:

        ### read a frame from the webcam
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        ### Prepare image for mediapipe processing in square shape
        ori_square_img = cvt_raw_image_squre(raw_image=raw_frame)
        mp_pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ori_square_img)
        limit_edge = ori_square_img.shape[0] # spuare image, so height and width are the same
    
        ### Pose detection
        detection_result = pose_detector.detect(mp_pose_image)
        if not detection_result.pose_landmarks:
            print("No pose landmarks detected in the image.")
            annotated_image = ori_square_img
        else:
            important_joints_coords = get_important_joints(
                img_height=ori_square_img.shape[0],
                img_width=ori_square_img.shape[1],
                detected_person=detection_result.pose_landmarks[0],
            )
            left_end_effector_coords = get_end_effector_coords(
                important_joints_coords=important_joints_coords,
                lr='left',
            )
            print(f"Left end effector coordinates: {left_end_effector_coords}")
            right_end_effector_coords = get_end_effector_coords(
                important_joints_coords=important_joints_coords,
                lr='right',
            )
            print(f"Right end effector coordinates: {right_end_effector_coords}")
            annotated_image = draw_pose_landmarks_on_image(
                original_image=ori_square_img,
                detection_result=detection_result,
            )
            ### Gesture recognition
            # left hand
            left_hand_image = get_hand_image(
                ori_square_img=ori_square_img,
                important_joints_coords=important_joints_coords,
                limit_edge=limit_edge,
                lr='left',
            )
            if left_hand_image is not None:
                mp_left_hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=left_hand_image)
                left_hand_recognition_result = gesture_recognizer.recognize(mp_left_hand_image)
                left_gesture = get_gesture(recognition_result=left_hand_recognition_result)
                print(f"Left gesture: {left_gesture}")
            else:
                left_gesture = None
                print("Left hand image is None, skipping gesture recognition for left hand.")

            # right hand
            right_hand_image = get_hand_image(
                ori_square_img=ori_square_img,
                important_joints_coords=important_joints_coords,
                limit_edge=limit_edge,
                lr='right',
            )
            if right_hand_image is not None:
                mp_right_hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=right_hand_image)
                right_hand_recognition_result = gesture_recognizer.recognize(mp_right_hand_image)
                right_gesture = get_gesture(recognition_result=right_hand_recognition_result)
                print(f"Right gesture: {right_gesture}")
            else:
                right_gesture = None
                print("Right hand image is None, skipping gesture recognition for left hand.")

        ### Display the annotated image
        cv2.imshow('Webcam', annotated_image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()

    