import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ✅ OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 불러오기
if not api_key:
    raise ValueError("🚨 OpenAI API 키가 설정되지 않았습니다. 환경 변수에 OPENAI_API_KEY를 설정하세요.")
os.environ["OPENAI_API_KEY"] = api_key

host = os.getenv("SERVER_HOST")
port = os.getenv("SERVER_PORT")

# ✅ 2개의 LLM 모델 설정
model_1 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 적합도 평가
model_2 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 첨삭 or 공부법 제공
model_3 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 인재유형 판단 모델

# ✅ 프롬프트 템플릿 정의
prompt_1 = PromptTemplate(
    input_variables=["lorem", "jobObjective"],
    template="""
    사용자가 입력한 자기소개서 내용이 {jobObjective}에 얼마나 적합한지 0~100% 사이의 점수로 평가하고,
    아래 형식으로만 답변해.
    
    적합도 점수: <숫자>%
    
    평가할 자기소개서:
    {lorem}
    """
)

prompt_2_resume = PromptTemplate(
    input_variables=["lorem", "jobObjective"],
    template="""
    사용자의 자기소개서를 {jobObjective}직무에 맞게 최적화해서 수정해주고, 400자 이상 작성해줘.
    사용자가 어떤 말투로 입력을 해도 너는 자기소개서 말투에 맞게 ~입니다처럼 작성해주고 되도록이면 STAR 기법을 사용해서 작성해주면 좋겠어
    
    원본:
    {lorem}
    
    수정된 자기소개서:
    """
)

prompt_2_study = PromptTemplate(
    input_variables=["jobObjective"],
    template="""
    사용자의 자기소개서가 {jobObjective}에 적합하지 않음.
    따라서 {jobObjective}에 맞는 실력을 키우기 위한 공부법과 방향을 제공해줘.
    
    추천 공부법:
    """
)

# ✅ 비동기 파이프라인 실행 함수로 변경
async def process_pipeline(lorem, jobObjective):
    # 1️⃣ 희망 직무 적합도 평가 (비동기 호출)
    response_1_obj = await (prompt_1 | model_1).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
    response_1 = response_1_obj.content
    response_1 = response_1.replace("적합도 점수: ", "").strip("%")
    print(f"🔹 적합도 평가 결과: {response_1}")

    # 2️⃣ 적합도가 75% 이상이면 자기소개서 첨삭, 아니면 공부법 추천
    if int(response_1) >= 75:
        response_2_obj = await (prompt_2_resume | model_2).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
        response_2 = response_2_obj.content
        return {"ability": response_1, "resume": response_2, "lorem": lorem}
    else:
        response_2_obj = await (prompt_2_study | model_2).ainvoke({"jobObjective": jobObjective})
        response_2 = response_2_obj.content
        return {"ability": response_1, "study": response_2, "lorem": lorem}

# ✅ FastAPI 설정
app = FastAPI()

# CORS 미들웨어 추가 (외부 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 특정 도메인만 허용 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class ResumeRequest(BaseModel):
    lorem: str
    jobObjective: str

# 엔드포인트: 비동기 함수로 선언하여 await process_pipeline 사용
@app.post("/user/validate_resume")
async def validate_resume(request: ResumeRequest):
    print("서버가 정상적으로 연결됐습니다.")
    lorem = request.lorem
    jobObjective = request.jobObjective 
    return await process_pipeline(lorem, jobObjective)

# ✅ 서버 실행 (다른 컴퓨터에서 접속 가능하도록 host와 port 지정)
if __name__ == "__main__":
    uvicorn.run("main:app", host=host, port=port, reload=True)
