# 실시간 오브젝트 트래킹


## 목표

개별적인 오브젝트를 실시간으로 CLASS NAME과 ID(개수)를 구분하면서 트래킹 수행 할 수 있어야 한다.<br>
즉, 화면상에 'Melona', 'BIBBIG'이 각각 1개, 3개로 보인다면 각각의 클래스 명과 개수를 구분하여 트래킹한다.
클래스명 'Melona'에 ID는 1이 부여된다면 'BIBBIG'의 ID는 2, 3, 4으로 정의되어야 하며 이것이 화면에서 사라지기 전까지는 그것은 그들의 고유한 ID로 정의되어야 한다.


##  ⚠️ 주의 & 문제점
- Re-ID <br>
ID switching의 문제가 있다. 다양한 객체를 추적할 때, 각 개체의 track ID가 바뀌는 현상이 존재한다.
<br>Re-identification 모델을 적용해야한다. 또는 OSNet 활용을 제안한다.


- custom tracking model training <br>


---
## ✅ 현재 사용 코드 (in vscode)
참고 : https://github.com/theos-ai/easy-yolov7#detect-the-webcam


### 환경
MAC OS [vscode] <br>
Python 3.9.18
<br><br>

### 시작에 앞서..
1. ```pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox``` 먼저 진행한다.
2. 'requirements.txt' 내용을 알맞게 수정한다.
#### 수정사항
- torch==2.0.0 <br>
- torchvision==0.15.0 <br>
- ~PyMuPDF==1.20.2~ # 삭제하세요. <br>
<br>

### 그외 에러 해결
#### Failed building wheel for PyMuPDF 문제 해결 (MAC)
- MAC에서는 터미널을 열고 다음을 수행한다. <br>
```brew install swig```






---

## 기존 일반 코드 (in vscode)

다운로드 : https://github.com/deshwalmahesh/yolov7-deepsort-tracking



```
from detection_helpers import *
from tracking_helpers import *
from bridge_wrapper import *
from PIL import Image

# vs code 기준
# 터미널 : pip install -r requirements.txt
# 폴더내 파일 수정 : data > coco.yaml 커스텀 내용대로 수정
detector = Detector(classes = [0,1])

# 이부분 수정해야 하는지 검토 필요 ',' <- 이부분
detector.load_model('best.pt',)

tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

#영상 기준
#tracker.track_video("10.mp4", output="./IO_data/output/11.mp4", show_live = False, skip_frames = 0, count_objects = True, verbose=1)

#실시간 웹캠 기준
tracker.track_video(0, output="./IO_data/output/webcam_output.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1) 
```
