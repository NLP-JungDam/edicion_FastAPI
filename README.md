RAG 방식을 사용한 Model 관련 FastAPI 입니다

가상환경 만들기

- python -m venv venv

가상환경 실행하기

- cd venv/scripts
- activate.bat
- cd..
- cd..
  (원래 환경으로 돌아오기)

이후부터 원하시는 라이브러리, 모듈 다운로드 후 사용하시면 됩니다.

requirement.txt에 있는 모듈 한번에 다운받기
(npm install 이라고 생각하시면 됩니다)

- pip install -r requirements.txt

main.py실행하기

- python main.py

uvicorn main:app을 사용해야하는거 아닌가?

- 만약 아래와 같은 코드가 없었다면 uvicorn main:app 'host IP입력' 'port Number입력' --reload를 사용해서 함
  - if **name** == "**main**" :
  - uvicorn.run("main:app", host=host, port=port, reload=True)
