import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


game_state = "title"
player_score = 0
computer_score = 0
match_toss = False
target = 0
waiting_for_release = False
first_batting = None
second_batting = None

# ==========================================
# COMPUTER IMAGE VARIABLES
# ==========================================

computer_image = None
computer_image_time = 0


def player_batting(player,computer):

    global player_score

    if player == computer:
        return False

    else:
        player_score += player
        return True


def computer_batting(player,computer):

    global computer_score

    if player == computer:
        return False

    else:
        computer_score += computer
        return True


def player_number():

    hand_size = math.sqrt(
        (hand_landmarks[0].x * w - l3[1][0]) ** 2 +
        (hand_landmarks[0].y * h - l3[1][1]) ** 2
    )

    distance_thumb = math.sqrt(
        (l1[4][0] - l3[0][0]) ** 2 +
        (l1[4][1] - l3[0][1]) ** 2
    )

    ratio = distance_thumb / hand_size

    finger_count = 0

    if angles[0] > 165:
        finger_count += 1

    if angles[1] > 165:
        finger_count += 1

    if angles[2] > 165:
        finger_count += 1

    if angles[3] > 165:
        finger_count += 1

    thumb_up = ratio > 0.60

    if finger_count == 4 and thumb_up:
        return 5

    if finger_count == 0 and thumb_up:
        return 6

    if finger_count >= 1:
        return finger_count

    return 0


def computer_number():

    a = random.randint(1,6)

    return a


def player_choice(n):

    if n >= 1:
        return "bat"

    else:
        return "bowl"


def computer_choice():

    return random.choice(['bat','bowl'])


# ==========================================
# RESET GAME
# ==========================================

def reset_game():

    global game_state
    global player_score
    global computer_score
    global match_toss
    global target
    global waiting_for_release
    global first_batting
    global second_batting
    global computer_image
    global computer_image_time

    game_state = "title"

    player_score = 0
    computer_score = 0

    match_toss = False

    target = 0

    waiting_for_release = False

    first_batting = None
    second_batting = None

    computer_image = None
    computer_image_time = 0


# ==========================================
# PLAY VIDEO
# ==========================================

def play_animation(video_path):

    animation = cv.VideoCapture(video_path)

    while animation.isOpened():

        ret, frame = animation.read()

        if not ret:
            break

        frame = cv.resize(frame, (1100, 700))

        cv.imshow("game", frame)

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    animation.release()


# ==========================================
# SHOW PHOTO
# ==========================================

def show_photo(photo_path,m):

    photo = cv.imread(photo_path)

    if photo is None:

        print("Could not load image")

        return

    photo = cv.resize(photo, (1100, 700))

    cv.imshow("game", photo)

    cv.waitKey(m)


# ==========================================
# TITLE SCREEN
# ==========================================

def title_screen():

    title = cv.imread("Photos\\title.png")

    rule = cv.imread("Photos\\RULES.png")

    title = cv.resize(title, (1100, 700))

    cv.imshow("game", title)

    a = 0

    while True:

        key = cv.waitKey(0) & 0xFF

        if key == ord('o'):

            rule = cv.resize(rule, (1100, 700))

            cv.imshow("game", rule)

            a = 1

        elif key == ord('q') and a == 1:

            cv.imshow("game", title)

            a = 0

        elif key == ord('q'):

            return


# ==========================================
# END SCREEN
# ==========================================

def end_screen(e):

    win = cv.imread(
        "Photos\\player_win_end_screen.png"
    )

    loose = cv.imread(
        "Photos\\computer_win_end_screen.png"
    )

    win = cv.resize(win, (1100, 700))

    loose = cv.resize(loose, (1100, 700))

    if e == 0:

        cv.imshow("game",win)

        while True:

            key = cv.waitKey(0) & 0xFF

            if key == ord('q'):

                return

    if e == 1:

        cv.imshow("game",loose)

        while True:

            key = cv.waitKey(0) & 0xFF

            if key == ord('q'):

                return


# ==========================================
# COMPUTER HAND SIGN
# ==========================================

def computer_hand_signs(n):

    global rgb_black

    rgb_black = cv.imread(
        f"Photos\\{n}.png"
    )

    rgb_black = cv.resize(
        rgb_black,
        (w,h)
    )


# ==========================================
# MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path = "hand_landmarker.task"
)


options = vision.HandLandmarkerOptions(

    base_options = base_options,

    num_hands = 1,

    min_hand_detection_confidence = 0.7,

    min_hand_presence_confidence = 0.7,

    min_tracking_confidence = 0.7
)


detector = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

cap = cv.VideoCapture(0)


# ==========================================
# CREATE ONE GAME WINDOW
# ==========================================

cv.namedWindow(
    "game",
    cv.WINDOW_NORMAL
)

cv.resizeWindow(
    "game",
    1100,
    700
)

cv.moveWindow(
    "game",
    150,
    50
)


# ==========================================
# MAIN GAME LOOP
# ==========================================

while True:

    ret,frame = cap.read()

    frame = cv.flip(frame,1)

    h,w,c = frame.shape

    rgb_frame = cv.cvtColor(
        frame,
        cv.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format = mp.ImageFormat.SRGB,

        data = rgb_frame
    )


    results = detector.detect(
        mp_image
    )


    # ==========================================
    # CREATE BLACK COMPUTER PANEL
    # ==========================================

    black = np.zeros(

        (
            frame.shape[0],
            frame.shape[1]
        ),

        dtype = 'uint8'
    )


    rgb_black = cv.cvtColor(

        black,
        cv.COLOR_BGR2RGB
    )


    # ==========================================
    # TITLE
    # ==========================================

    if game_state == "title":

        title_screen()

        show_photo(
            "Photos\\TOSS.png",
            2000
        )

        game_state = "odd_or_even"


    # ==========================================
    # COMPUTER CHOICE
    # ==========================================

    elif game_state == "computer_choice":

        choice = computer_choice()

        if choice == "bat":

            game_state = "computer_batting"

            print(
                "computer choose batting"
            )

            show_photo(
                "Photos\\computer_batting.png",
                2000
            )

            first_batting = "computer"

            second_batting = "player"

            waiting_for_release = True


        elif choice == "bowl":

            game_state = "player_batting"

            print(
                "computer choose bowling"
            )

            show_photo(
                "Photos\\computer_bowling.png",
                2000
            )

            first_batting = "player"

            second_batting = "computer"

            waiting_for_release = True


    # ==========================================
    # HAND DETECTION
    # ==========================================

    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:


            connections = [

                (0, 1), (1, 2), (2, 3), (3,4),

                (0, 5), (5, 6), (6, 7), (7, 8),

                (5, 9), (9, 10), (10, 11), (11, 12),

                (9, 13), (13, 14), (14, 15), (15, 16),

                (13, 17), (17,18), (18,19), (19,20),

                (0, 17)

            ]


            # ==========================================
            # LANDMARKS
            # ==========================================

            for landmarks in hand_landmarks:

                px = int(
                    landmarks.x * w
                )

                py = int(
                    landmarks.y * h
                )

                cv.circle(

                    frame,

                    (px,py),

                    7,

                    (0,255,0),

                    -1
                )


            # ==========================================
            # CONNECTIONS
            # ==========================================

            for start,end in connections:

                x1 = int(
                    hand_landmarks[start].x*w
                )

                y1 = int(
                    hand_landmarks[start].y*h
                )

                x2 = int(
                    hand_landmarks[end].x*w
                )

                y2 = int(
                    hand_landmarks[end].y*h
                )

                cv.line(

                    frame,

                    (x1,y1),

                    (x2,y2),

                    (0,0,255),

                    2
                )


            # ==========================================
            # L1
            # ==========================================

            l1 = [

                (
                    hand_landmarks[8].x * w,
                    hand_landmarks[8].y * h
                ),

                (
                    hand_landmarks[12].x * w,
                    hand_landmarks[12].y * h
                ),

                (
                    hand_landmarks[16].x * w,
                    hand_landmarks[16].y * h
                ),

                (
                    hand_landmarks[20].x * w,
                    hand_landmarks[20].y * h
                ),

                (
                    hand_landmarks[4].x * w,
                    hand_landmarks[4].y * h
                )

            ]


            # ==========================================
            # L2
            # ==========================================

            l2 = [

                (
                    hand_landmarks[6].x * w,
                    hand_landmarks[6].y * h
                ),

                (
                    hand_landmarks[10].x * w,
                    hand_landmarks[10].y * h
                ),

                (
                    hand_landmarks[14].x * w,
                    hand_landmarks[14].y * h
                ),

                (
                    hand_landmarks[18].x * w,
                    hand_landmarks[18].y * h
                ),

                (
                    hand_landmarks[3].x * w,
                    hand_landmarks[3].y * h
                )

            ]


            # ==========================================
            # L3
            # ==========================================

            l3 = [

                (
                    hand_landmarks[5].x * w,
                    hand_landmarks[5].y * h
                ),

                (
                    hand_landmarks[9].x * w,
                    hand_landmarks[9].y * h
                ),

                (
                    hand_landmarks[13].x * w,
                    hand_landmarks[13].y * h
                ),

                (
                    hand_landmarks[17].x * w,
                    hand_landmarks[17].y * h
                ),

                (
                    hand_landmarks[2].x * w,
                    hand_landmarks[2].y * h
                )

            ]


            # ==========================================
            # U
            # ==========================================

            u = [

                (
                    l1[0][0] - l2[0][0],
                    l1[0][1] - l2[0][1]
                ),

                (
                    l1[1][0] - l2[1][0],
                    l1[1][1] - l2[1][1]
                ),

                (
                    l1[2][0] - l2[2][0],
                    l1[2][1] - l2[2][1]
                ),

                (
                    l1[3][0] - l2[3][0],
                    l1[3][1] - l2[3][1]
                ),

                (
                    l1[4][0] - l2[4][0],
                    l1[4][1] - l2[4][1]
                )

            ]


            # ==========================================
            # V
            # ==========================================

            v = [

                (
                    l3[0][0] - l2[0][0],
                    l3[0][1] - l2[0][1]
                ),

                (
                    l3[1][0] - l2[1][0],
                    l3[1][1] - l2[1][1]
                ),

                (
                    l3[2][0] - l2[2][0],
                    l3[2][1] - l2[2][1]
                ),

                (
                    l3[3][0] - l2[3][0],
                    l3[3][1] - l2[3][1]
                ),

                (
                    l3[4][0] - l2[4][0],
                    l3[4][1] - l2[4][1]
                )

            ]


            # ==========================================
            # MAG U
            # ==========================================

            mag_u = [

                math.sqrt(
                    pow(u[0][0],2) +
                    pow(u[0][1],2)
                ),

                math.sqrt(
                    pow(u[1][0],2) +
                    pow(u[1][1],2)
                ),

                math.sqrt(
                    pow(u[2][0],2) +
                    pow(u[2][1],2)
                ),

                math.sqrt(
                    pow(u[3][0],2) +
                    pow(u[3][1],2)
                ),

                math.sqrt(
                    pow(u[4][0],2) +
                    pow(u[4][1],2)
                )

            ]


            # ==========================================
            # MAG V
            # ==========================================

            mag_v = [

                math.sqrt(
                    pow(v[0][0],2) +
                    pow(v[0][1],2)
                ),

                math.sqrt(
                    pow(v[1][0],2) +
                    pow(v[1][1],2)
                ),

                math.sqrt(
                    pow(v[2][0],2) +
                    pow(v[2][1],2)
                ),

                math.sqrt(
                    pow(v[3][0],2) +
                    pow(v[3][1],2)
                ),

                math.sqrt(
                    pow(v[4][0],2) +
                    pow(v[4][1],2
                ))

            ]


            # ==========================================
            # DOT PRODUCT
            # ==========================================

            dot_product = [

                u[0][0] * v[0][0] +
                u[0][1] * v[0][1],

                u[1][0] * v[1][0] +
                u[1][1] * v[1][1],

                u[2][0] * v[2][0] +
                u[2][1] * v[2][1],

                u[3][0] * v[3][0] +
                u[3][1] * v[3][1],

                u[4][0] * v[4][0] +
                u[4][1] * v[4][1]

            ]


            # ==========================================
            # ANGLES
            # ==========================================

            angles = [

                math.degrees(
                    math.acos(
                        dot_product[0] /
                        (mag_u[0] * mag_v[0])
                    )
                ),

                math.degrees(
                    math.acos(
                        dot_product[1] /
                        (mag_u[1] * mag_v[1])
                    )
                ),

                math.degrees(
                    math.acos(
                        dot_product[2] /
                        (mag_u[2] * mag_v[2])
                    )
                ),

                math.degrees(
                    math.acos(
                        dot_product[3] /
                        (mag_u[3] * mag_v[3])
                    )
                ),

                math.degrees(
                    math.acos(
                        dot_product[4] /
                        (mag_u[4] * mag_v[4])
                    )
                )

            ]


            # ==========================================
            # GAME LOGIC
            # ==========================================

            if not waiting_for_release:


                # ======================================
                # ODD / EVEN
                # ======================================

                if game_state == "odd_or_even":

                    player = player_number()

                    if player == 1:

                        b = 1

                        print(
                            "player choose tails"
                        )

                    else:

                        b = 2

                        print(
                            "player choose heads"
                        )

                    game_state = "toss"


                # ======================================
                # TOSS
                # ======================================

                elif game_state == "toss":

                    toss_result = random.randint(1,2)


                    if toss_result == 1:

                        play_animation(
                            "Videos\\tails.mp4"
                        )

                        if b == 1:

                            print(
                                "player won toss"
                            )

                            show_photo(
                                "Photos\\player_won_toss.png",
                                2000
                            )

                            waiting_for_release = True

                            player_won = True

                        else:

                            print(
                                "player lost toss"
                            )

                            show_photo(
                                "Photos\\computer_won_toss.png",
                                2000
                            )

                            player_won = False


                    elif toss_result == 2:

                        play_animation(
                            "Videos\\heads.mp4"
                        )

                        if b == 2:

                            print(
                                "player won toss"
                            )

                            show_photo(
                                "Photos\\player_won_toss.png",
                                2000
                            )

                            waiting_for_release = True

                            player_won = True

                        else:

                            print(
                                "player lost toss"
                            )

                            show_photo(
                                "Photos\\computer_won_toss.png",
                                2000
                            )

                            player_won = False


                    if player_won:

                        game_state = "bat_or_bowl"

                    else:

                        game_state = "computer_choice"


                # ======================================
                # BAT OR BOWL
                # ======================================

                elif game_state == "bat_or_bowl":

                    if player_won:

                        player = player_number()

                        print(player)

                        choice = player_choice(player)


                        if choice == "bat":

                            game_state = "player_batting"

                            waiting_for_release = True

                            first_batting = "player"

                            second_batting = "computer"

                            print(
                                "player choose batting"
                            )

                            show_photo(
                                "Photos\\player_batting.png",
                                2000
                            )


                        elif choice == "bowl":

                            game_state = "computer_batting"

                            waiting_for_release = True

                            print(
                                "player choose bowling"
                            )

                            show_photo(
                                "Photos\\player_bowling.png",
                                2000
                            )

                            first_batting = "computer"

                            second_batting = "player"


                    else:

                        game_state = "computer_choice"


                # ======================================
                # PLAYER BATTING
                # ======================================

                elif game_state == "player_batting":


                    if first_batting == "player":

                        player = player_number()

                        computer = computer_number()


                        # COMPUTER IMAGE
                        computer_image = cv.imread(
                            f"Photos\\{computer}.png"
                        )

                        computer_image = cv.resize(
                            computer_image,
                            (w,h)
                        )

                        # START TIMER
                        computer_image_time = cv.getTickCount()


                        print(
                            f"player: {player}"
                        )

                        print(
                            f"computer: {computer}"
                        )


                        if player_batting(
                            player,
                            computer
                        ):

                            waiting_for_release = True

                        else:

                            print("out!")

                            print(player_score)

                            game_state = "computer_batting"

                            target = player_score+1

                            print(
                                f"target = {target}"
                            )

                            waiting_for_release = True


                    elif first_batting == "computer":

                        player = player_number()

                        computer = computer_number()


                        # COMPUTER IMAGE
                        computer_image = cv.imread(
                            f"Photos\\{computer}.png"
                        )

                        computer_image = cv.resize(
                            computer_image,
                            (w,h)
                        )

                        # START TIMER
                        computer_image_time = cv.getTickCount()


                        print(
                            f"player: {player}"
                        )

                        print(
                            f"computer: {computer}"
                        )


                        if player_batting(
                            player,
                            computer
                        ):

                            if player_score >= target:

                                print(
                                    "player win!"
                                )

                                game_state = "title"

                                end_screen(0)

                                # RESET AFTER MATCH
                                reset_game()

                            waiting_for_release = True


                        else:

                            print("out!")

                            print(player_score)

                            game_state = "title"

                            print(
                                "computer win"
                            )

                            end_screen(1)

                            # RESET AFTER MATCH
                            reset_game()


                # ======================================
                # COMPUTER BATTING
                # ======================================

                elif game_state == "computer_batting":


                    if first_batting == "computer":

                        player = player_number()

                        computer = computer_number()


                        # COMPUTER IMAGE
                        computer_image = cv.imread(
                            f"Photos\\{computer}.png"
                        )

                        computer_image = cv.resize(
                            computer_image,
                            (w,h)
                        )

                        # START TIMER
                        computer_image_time = cv.getTickCount()


                        print(
                            f"player: {player}"
                        )

                        print(
                            f"computer: {computer}"
                        )


                        if computer_batting(
                            player,
                            computer
                        ):

                            waiting_for_release = True

                        else:

                            print("out!")

                            print(computer_score)

                            waiting_for_release = True

                            game_state = "player_batting"

                            target = computer_score+1

                            print(
                                f"target = {target}"
                            )


                    elif first_batting == "player":

                        player = player_number()

                        computer = computer_number()


                        # COMPUTER IMAGE
                        computer_image = cv.imread(
                            f"Photos\\{computer}.png"
                        )

                        computer_image = cv.resize(
                            computer_image,
                            (w,h)
                        )

                        # START TIMER
                        computer_image_time = cv.getTickCount()


                        print(
                            f"player: {player}"
                        )

                        print(
                            f"computer: {computer}"
                        )


                        if computer_batting(
                            player,
                            computer
                        ):

                            if computer_score >= target:

                                print(
                                    "computer win!"
                                )

                                end_screen(1)

                                game_state = "title"

                                # RESET AFTER MATCH
                                reset_game()


                            waiting_for_release = True


                        else:

                            print("out!")

                            print(computer_score)

                            print("player win")

                            end_screen(0)

                            game_state = "title"

                            # RESET AFTER MATCH
                            reset_game()


    else:

        waiting_for_release = False


    # ==========================================
    # PLAYER SCORE
    # ==========================================

    cv.putText(

        frame,

        f"Score: {player_score}",

        (20,40),

        cv.FONT_HERSHEY_SIMPLEX,

        1,

        (0,0,0),

        2
    )


    # ==========================================
    # TARGET
    # ==========================================

    cv.putText(

        frame,

        f"Target: {target}",

        (20,80),

        cv.FONT_HERSHEY_SIMPLEX,

        1,

        (0,0,0),

        2
    )


    # ==========================================
    # COMPUTER IMAGE TIMER
    # ==========================================

    if computer_image is not None:

        elapsed_time = (

            cv.getTickCount()
            -
            computer_image_time

        ) / cv.getTickFrequency()


        if elapsed_time < 1.5:

            computer_panel = computer_image

        else:

            computer_panel = rgb_black

    else:

        computer_panel = rgb_black


    # ==========================================
    # COMPUTER SCORE
    # ==========================================

    cv.putText(

        computer_panel,

        f"Score: {computer_score}",

        (20,40),

        cv.FONT_HERSHEY_SIMPLEX,

        1,

        (255,255,255),

        2
    )


    # ==========================================
    # SPLIT SCREEN
    # ==========================================

    screen = np.hstack(

        (
            frame,
            computer_panel
        )

    )


    cv.imshow(
        "game",
        screen
    )


    # ==========================================
    # QUIT
    # ==========================================

    if cv.waitKey(1) & 0xff == ord('q'):

        break


# ==========================================
# CLEAN UP
# ==========================================

cap.release()

cv.destroyAllWindows()