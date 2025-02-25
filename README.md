RAG 방식을 사용한 Model 관련 FastAPI 입니다

가상환경 만들기

-   python -m venv venv

가상환경 실행하기

-   cd venv/scripts
-   activate.bat
-   cd..
-   cd..
    (원래 환경으로 돌아오기)

이후부터 원하시는 라이브러리, 모듈 다운로드 후 사용하시면 됩니다.

requirement.txt에 있는 모듈 한번에 다운받기
(npm install 이라고 생각하시면 됩니다)

-   pip install -r requirements.txt

main.py실행하기

-   python main.py

uvicorn main:app을 사용해야하는거 아닌가?

-   만약 아래와 같은 코드가 없었다면 uvicorn main:app 'host IP입력' 'port Number입력' --reload를 사용해서 함
    -   if **name** == "**main**" :
    -   uvicorn.run("main:app", host=host, port=port, reload=True)

※ 경고
저는 python 3.11버전으로 가상환경 구축하여 진행하였습니다.
다른 로컬 환경에서 가상환경 만들고 실행해가며 requirements.txt 수정 중입니다!
pip와 python 버전이 다르면 오류가 나는것을 확인하였습니다

pip와 python 버전 확인하는 법

-   pip -V
    (저는 pip==24.0, python==3.11 입니다)

Warning이 뜨는 경우

-   pip install -U langchain_chroma langchain_openai
이후 업데이트에 따라 document 확인해주셔야합니다!