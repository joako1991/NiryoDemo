"""
This script shows an example of how to use Ned's vision to
make a conditioning with any of the objects supplied with the Vision Kit
The script works in 2 ways:
- One where all the vision process is made on the robot
- One where the vision process is made on the computer

The first one shows how easy it is to use Ned's vision kit with PyNiryo
The second demonstrates a way to do image processing from user's computer. It highlights
the fact that the user can imagine every type of process on his computer.

The objects will be conditioned in a grid of dimension grid_dimension. If the grid is completed,
objects will be pack over the lower level
"""

from pyniryo2 import *
import pyniryo
import cv2
from utils import *
from aruco import ArucoDetector

# -- MUST Change these variables
robot_ip_address = "10.131.12.8"  # IP address of Ned
workspace_name = "paper_workspace"

def init_robot(ip_address):
    # Connect to robot
    robot = NiryoRobot(ip_address)

    # Changing tool
    robot.tool.update_tool()

    # Calibrate robot if robot needs calibration
    robot.arm.calibrate_auto()

    return robot

def go_to_observation_point(robot):
    # -- Should Change these variables
    # The pose from where the image processing happens
    # Joints space
    observation_pose = [1.57, 0.27, -0.31, 0.0, -1.57, 0.0]

    # Moving to observation pose
    robot.arm.move_joints(observation_pose)

def grasp_operation(robot, pick_pose, height_offset_meters=0.1):
    robot.tool.release_with_tool()
    robot.arm.move_pose(pick_pose)

    go_back_pose = pick_pose
    go_back_pose.z += height_offset_meters
    # niryo_robot.pick_place.pick_from_pose(obj_pose)
    robot.tool.close_gripper(hold_torque_percentage=100)
    robot.arm.move_pose(go_back_pose)

def go_to_release_pose(robot):
    place_point_joints = objects.PoseObject(
        x = -0.0023,
        y = -0.276,
        z = 0.303,
        roll = 0.0,
        pitch = 0.881,
        yaw = 3.139,
    )

    # Placing
    robot.arm.move_pose(place_point_joints)

def release_operation(robot, tag_detector):
    robot.tool.open_gripper()

def get_robot_image(robot):
    mtx, dist = robot.vision.get_camera_intrinsics()
    img_compressed = robot.vision.get_img_compressed()
    img = uncompress_image(img_compressed)
    img = undistort_image(img, mtx, dist)
    return img

def detect_pieces(robot, raw_img):
    obj_pose = None
    obj_found = False
    work_space_img = None
    thresholded_img = None

    # extracting working area
    work_space_img = pyniryo.extract_img_workspace(raw_img, workspace_ratio=1.0)

    if work_space_img is not None:
        # Applying Threshold on ObjectColor
        color_hsv_setting = ColorHSV.ANY.value
        thresholded_img = threshold_hsv(work_space_img, *color_hsv_setting)

        # Getting biggest contour/blob from threshold image
        contour = biggest_contour_finder(thresholded_img)
        if contour is None or len(contour) == 0:
            print("No blob found")
            obj_found = False

        else:
            # Getting contour / blob center and angle
            cx, cy = get_blob_barycenter(contour)

            cx_rel, cy_rel = relative_pos_from_pixels(work_space_img, cx, cy)
            angle = get_contour_angle(contour)

            # Getting object world pose from relative pose
            ## All the arguments are in meters
            obj_pose = robot.vision.get_target_pose_from_rel(workspace_name,
                                                            height_offset=0.008,
                                                            x_rel=cx_rel, y_rel=cy_rel,
                                                            yaw_rel=angle)
            obj_found = True

    return obj_found, obj_pose, work_space_img, thresholded_img

def show_stream(raw, thres_img):
    if raw is not None:
        cv2.namedWindow('Nyrio img', cv2.WINDOW_NORMAL)
        cv2.imshow('Nyrio img', raw)
        cv2.namedWindow('Thresholded img', cv2.WINDOW_NORMAL)
        cv2.imshow('Thresholded img', thres_img)

        cv2.waitKey(20)

def main():
    niryo_robot = init_robot(robot_ip_address)
    # Initializing variables
    obj_pose = None
    is_running = True

    mtx, dist = niryo_robot.vision.get_camera_intrinsics()
    tag_det = ArucoDetector(mtx, dist)

    while is_running:
        # Moving to observation pose
        go_to_observation_point(niryo_robot)
        img = get_robot_image(niryo_robot)

        key = cv2.waitKey(20)
        if key == ord('q') or key == ord('Q'):
            is_running = False
            continue

        obj_found, obj_pose, work_space_img, thresholded_img = detect_pieces(niryo_robot, img)

        show_stream(work_space_img, thresholded_img)

        if not obj_found:
            print("Unable to find markers")
            continue
        else:
            # Everything is good, so we going to object
            grasp_operation(niryo_robot, obj_pose, height_offset_meters=0.15)
            go_to_release_pose(niryo_robot)
            release_operation(niryo_robot, tag_det)

    niryo_robot.end()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
