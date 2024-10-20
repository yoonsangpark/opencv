import cv2
import numpy as np
from time import sleep

# https://github.com/dltpdn/insightbook.opencv_project_python/blob/master/04.img_processing/workshop_cctv_motion_sensor.py
# 감도 설정(카메라 품질에 따라 조정 필요)
thresh = 25    # 달라진 픽셀 값 기준치 설정
max_diff = 50   # 달라진 픽셀 갯수 기준치 설정

# 동영상 장치 준비
a, b = None, None

video_file ='C:\\yonsangpark\\opencv\\KakaoTalk_20240922_130937638.mp4'
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
print("FPS : %f, Delay : %d ms" %(fps, delay))

total_frmaes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("total_frmaes : %d " %(total_frmaes))

#start frame
srt_idx = fps * 36

if cap.isOpened():
    frm_idx = srt_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
    ret, a = cap.read()         # a 프레임 읽기
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    while ret:
        frm_idx += fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)        
        ret, b = cap.read()     # c 프레임 읽기
                
        if not ret:
            break

        draw = b.copy()         # 출력 영상에 사용할 복제본
        
        # 영상을 그레이 스케일로 변경        
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        # a-b 절대 값 차 구하기 
        diff1 = cv2.absdiff(a_gray, b_gray)

        # 스레시홀드로 기준치 이내의 차이는 무시
        ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)

        # 두 차이에 대해서 AND 연산, 두 영상의 차이가 모두 발견된 경우
        #diff = cv2.bitwise_and(diff1_t, diff2_t)

        # 열림 연산으로 노이즈 제거 ---①
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (40,40))
        diff = cv2.morphologyEx(diff1_t, cv2.MORPH_OPEN, k)

        # 차이가 발생한 픽셀이 갯수 판단 후 사각형 그리기
        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            nzero = np.nonzero(diff)  # 0이 아닌 픽셀의 좌표 얻기(y[...], x[...])
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])), \
                                (max(nzero[1]), max(nzero[0])), (0,255,0), 2)
            cv2.putText(draw, "Motion Detected", (10,30), \
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255))
        
        # 컬러 스케일 영상과 스레시홀드 영상을 통합해서 출력
        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion sensor',stacked )

        sleep(1)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
cap.release()
cv2.destroyAllWindows()
