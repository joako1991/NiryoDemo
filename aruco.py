import cv2

class ArucoDetector(object):
    def __init__(self, intrinsics, dist_params):
        self.dictionary_id = cv2.aruco.DICT_6X6_50
        self.intrinsics = intrinsics
        self.dist_params = dist_params

    def detect_and_draw_marker(self, img):
        arucoDict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        arucoParams = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(
            img,
            arucoDict,
            parameters=arucoParams)

        if len(corners) > 0:
            for markerCorner, _ in zip(corners, ids.flatten()):
                corners_aruco = markerCorner.reshape((4, 2))
                (topLeft, _, bottomRight, _) = corners_aruco

                cv2.polylines(img, [markerCorner.astype(int)], True, (0, 255, 0), 2)

                cX = int((topLeft[0] + bottomRight[0]) / 2)
                cY = int((topLeft[1] + bottomRight[1]) / 2)

                cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
        return img

    def pose_estimation(self, img):
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        rvec = []
        tvec = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i],
                    0.025,
                    self.intrinsics,
                    self.dist_params)

                cv2.aruco.drawDetectedMarkers(img, corners)
                cv2.drawFrameAxes(img,
                    self.intrinsics,
                    self.dist_params,
                    rvec,
                    tvec,
                    0.01)

        n_ids = 0
        if ids is not None:
            n_ids = len(ids)
        return img, n_ids, rvec, tvec
