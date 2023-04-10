import cv2
import mediapipe as mp
import numpy as np
import time, os


# 3개 제스쳐 모을겨
actions = ['one', 'two', 'three']
# 윈도우 사이즈 30
seq_length = 30
# 액션 이 시간동안 녹화 
motion_time = 8;

# (첫 카메라 렉 졸라걸려서 첫번째 모션 녹화 시간 늘림)
secs_for_action = 20

# 문서 이름으로 저장해보자
name = "secret1"

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# 캡으로 열었다
cap = cv2.VideoCapture(0)


# 타이머
created_time = int(time.time())


# dataset 폴더 만들겨
os.makedirs('dataset', exist_ok=True)


# 카메라 켜졌을 때
while cap.isOpened():
    # 위에 제스쳐 배열 동안 반복할겨
    for idx, action in enumerate(actions):
        data = []


        # 이미지 가져올겨
        ret, img = cap.read()


        # 플립 시킬겨. 셀카라 좌우반전인듯 ㅋㅋ
        img = cv2.flip(img, 1)



        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # 창 열어
        cv2.imshow('img', img)
        # 5초동안 대기해줌
        cv2.waitKey(5000)
        #시간 초기화
        start_time = time.time()




        # 위 설정 시간(secs_for_action)동안 반복
        while time.time() - start_time < secs_for_action:
            # 이미지 읽어와
            ret, img = cap.read()

            #플립해
            img = cv2.flip(img, 1)
            # 색깔 바꾸는겨
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 결과를 미디어파이프에 넣는겨
            result = hands.process(img)
            # 색깔 바꾸는겨
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 손 있으면
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        # x, y, z좌표 + visibility까지(이미지 상에서 보이는지 안보이는지) 체크
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 손 관절들 각도 계산하는겨
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    angle = np.degrees(angle) # Convert radian to degree

                    # 라벨 달아주는겨. idx는 제스처 배열의 인덱스
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    # joint(x, y, z, visibility)를 펼쳐가지고 이어줘 -> 100개짜리 행렬
                    d = np.concatenate([joint.flatten(), angle_label])

                    # data 변수에 append
                    data.append(d)

                    # 랜드마크 그려줘
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break


        # dat를 numpy array 형태로 변환
        data = np.array(data)
        # 프린트
        print(action, data.shape)
        # 아까 만든 dataset 폴더에 npy 형태 데이터로 저장
        np.save(os.path.join('dataset', f'raw_{name}_{action}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            # full_seq_data 배열에 한 스텝씩 넘어가면서 차근차근 저장하는겨
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        
        # full_seq_data배열을 seq 이름 붙혀서 저장. seq 데이터들 사용해서 학습할것
        np.save(os.path.join('dataset', f'seq_{name}_{action}'), full_seq_data)
        
        # 액션 이 시간동안 녹화 (첫 카메라 렉 -> 배열 하나 안나옴 -> 에러 때매 처음에만 늘림)
        secs_for_action = motion_time
    break