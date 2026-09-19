## Learning Purpose
본 프로젝트는 한국어 자연어처리(NLP), 문장 임베딩, 비지도 군집화 및 데이터 시각화 기술을 학습하고 실제 행정 데이터 분석 시나리오에 적용해보기 위해 제작한 개인 학습 프로젝트​입니다.
포함된 민원 데이터는 테스트를 위해 AI를 통해 생성된 가상 데이터입니다.


## How to Use
```
git clone https://github.com/poris551/civil-complaint-ai.git
cd civil-complaint-ai
pip install -r requirements.txt
python -m streamlit run app.py
```

실행 후 브라우저에서 민원 분석 대시보드를 확인할 수 있습니다.

KoBERT 분류 모델을 학습: python train_kobert.py

테스트 데이터는 data/test_complaints.csv에 포함되어 있습니다.
