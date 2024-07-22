from openpose.detect import *

if __name__ == "__main__":
    model = Body('body_pose_model.pth')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidate, subset = model(frame)
        canvas = draw_bodypose(frame, candidate, subset)
        cv2.imshow('Body Pose Detection', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()