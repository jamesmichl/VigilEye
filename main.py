import cv2

import config
from alarm import AlarmManager
from detector import DrowsinessDetector


def draw_landmarks(
    frame,
    points,
    color=(0, 255, 0)
):
    for point in points:
        cv2.circle(
            frame,
            point,
            2,
            color,
            -1
        )


def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.6,
    thickness=2
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_status_color(status):

    if status == "NORMAL":
        return (0, 255, 0)

    if status == "WARNING":
        return (0, 255, 255)

    if status == "DROWSY":
        return (0, 0, 255)

    return (255, 255, 255)


def main():

    detector = DrowsinessDetector()
    alarm = AlarmManager()

    camera = cv2.VideoCapture(
        config.CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "[ERROR] Could not open camera."
        )

        detector.close()
        alarm.cleanup()

        return

    print("=" * 50)
    print("DROWSINESS DETECTION SYSTEM")
    print("=" * 50)

    print(
        "Look straight at the camera and keep "
        "your eyes naturally open during calibration."
    )

    print()
    print("Controls:")
    print("Q = Quit")
    print("R = Recalibrate")
    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print(
                    "[ERROR] Failed to read camera frame."
                )
                break

            # Mirror webcam
            frame = cv2.flip(
                frame,
                1
            )

            data = detector.process(
                frame
            )

            height, width = frame.shape[:2]

            # FACE NOT DETECTED

            if not data["face_detected"]:

                alarm.stop()

                draw_text(
                    frame,
                    "FACE NOT DETECTED",
                    (30, 50),
                    (0, 0, 255),
                    1.0,
                    3
                )

                draw_text(
                    frame,
                    "Please face the camera",
                    (30, 85),
                    (255, 255, 255)
                )

            # FACE DETECTED

            else:

                # Eye landmarks
                draw_landmarks(
                    frame,
                    data["left_eye"]
                )

                draw_landmarks(
                    frame,
                    data["right_eye"]
                )

                # Mouth landmarks
                draw_landmarks(
                    frame,
                    data["mouth"],
                    (255, 0, 255)
                )

                # CALIBRATION
                if not data["calibrated"]:

                    alarm.stop()

                    progress = int(
                        data[
                            "calibration_progress"
                        ] * 100
                    )

                    draw_text(
                        frame,
                        "CALIBRATING",
                        (30, 50),
                        (0, 255, 255),
                        1.0,
                        3
                    )

                    draw_text(
                        frame,
                        "Look straight and keep eyes open naturally",
                        (30, 85)
                    )

                    draw_text(
                        frame,
                        f"Progress: {progress}%",
                        (30, 120)
                    )

                    draw_text(
                        frame,
                        f"EAR: {data['ear']:.3f}",
                        (30, 155)
                    )

                    # Progress bar
                    bar_width = 300

                    cv2.rectangle(
                        frame,
                        (30, 175),
                        (
                            30 + bar_width,
                            195
                        ),
                        (255, 255, 255),
                        2
                    )

                    cv2.rectangle(
                        frame,
                        (30, 175),
                        (
                            30
                            + int(
                                bar_width
                                * data[
                                    "calibration_progress"
                                ]
                            ),
                            195
                        ),
                        (0, 255, 255),
                        -1
                    )

                # DETECTION

                else:

                    status = data["status"]

                    color = get_status_color(
                        status
                    )

                    if status == "DROWSY":
                        alarm.play()
                    else:
                        alarm.stop()

                    draw_text(
                        frame,
                        f"STATUS: {status}",
                        (30, 50),
                        color,
                        1.0,
                        3
                    )

                    draw_text(
                        frame,
                        f"EAR: {data['ear']:.3f}",
                        (30, 90)
                    )

                    draw_text(
                        frame,
                        (
                            "Relative EAR: "
                            f"{data['relative_ear'] * 100:.1f}%"
                        ),
                        (30, 120)
                    )

                    draw_text(
                        frame,
                        (
                            "Threshold: "
                            f"{data['threshold']:.3f}"
                        ),
                        (30, 150)
                    )

                    draw_text(
                        frame,
                        (
                            "Eye closed: "
                            f"{data['closed_duration']:.1f}s"
                        ),
                        (30, 180)
                    )

                    draw_text(
                        frame,
                        (
                            "PERCLOS: "
                            f"{data['perclos'] * 100:.1f}%"
                        ),
                        (30, 210)
                    )

                    draw_text(
                        frame,
                        f"MAR: {data['mar']:.3f}",
                        (30, 240)
                    )

                    yawn_text = (
                        "YES"
                        if data["yawn"]
                        else "NO"
                    )

                    draw_text(
                        frame,
                        f"Yawn: {yawn_text}",
                        (30, 270)
                    )

                    draw_text(
                        frame,
                        (
                            "Yawn Count: "
                            f"{data['yawn_count']}"
                        ),
                        (30, 300)
                    )

                    draw_text(
                        frame,
                        (
                            "Drowsiness Score: "
                            f"{data['score']}"
                        ),
                        (30, 330),
                        color
                    )

                    # DROWSY WARNING

                    if status == "DROWSY":

                        overlay = frame.copy()

                        cv2.rectangle(
                            overlay,
                            (0, 0),
                            (width, height),
                            (0, 0, 255),
                            20
                        )

                        frame = cv2.addWeighted(
                            overlay,
                            0.4,
                            frame,
                            0.6,
                            0
                        )

                        text = "DROWSINESS DETECTED!"

                        text_size = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            3
                        )[0]

                        x = (
                            width
                            - text_size[0]
                        ) // 2

                        draw_text(
                            frame,
                            text,
                            (
                                x,
                                height - 70
                            ),
                            (0, 0, 255),
                            1.2,
                            3
                        )

                        wake_text = "WAKE UP!"

                        wake_size = cv2.getTextSize(
                            wake_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            3
                        )[0]

                        wake_x = (
                            width
                            - wake_size[0]
                        ) // 2

                        draw_text(
                            frame,
                            wake_text,
                            (
                                wake_x,
                                height - 30
                            ),
                            (0, 0, 255),
                            1.0,
                            3
                        )

            # CONTROLS

            draw_text(
                frame,
                "Q: Quit | R: Recalibrate",
                (
                    20,
                    height - 10
                ),
                (200, 200, 200),
                0.45,
                1
            )

            cv2.imshow(
                config.WINDOW_NAME,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord("r"):

                alarm.stop()

                detector.reset_calibration()

    except KeyboardInterrupt:

        print(
            "\nProgram interrupted."
        )

    finally:

        alarm.cleanup()

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

        print(
            "Drowsiness Detection System stopped."
        )


if __name__ == "__main__":
    main()