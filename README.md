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