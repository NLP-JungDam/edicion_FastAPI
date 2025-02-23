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
port = int(os.getenv("SERVER_PORT"))

# ✅ 2개의 LLM 모델 설정
model_1 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 적합도 평가
model_2 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 첨삭 or 공부법 제공
model_3 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 인재유형 판단 모델
model_4 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")  # 기업과 
 
# ✅ 프롬프트 템플릿 정의
prompt_1 = PromptTemplate(
    input_variables=["lorem", "jobObjective"],
    template="""
    당신은 아래의 10개 직업 카테고리에 대해, 주어진 자기소개서가 얼마나 적합한지 각각 0~100% 사이의 점수를 매기는 평가자입니다.

    직업 카테고리:
    1) 서비스업
    2) 제조·화학
    3) IT·웹·통신
    4) 은행·금융업
    5) 미디어·디자인
    6) 교육업
    7) 의료·제약·복지
    8) 판매·유통
    9) 건설업
    10) 기관·협회

    사용자가 선택한 직업 목표: {jobObjective}

    평가 기준:
    1. 먼저, 자기소개서가 {jobObjective} 직업에 얼마  나 적합한지 0~100% 사이의 점수로 평가하세요.
    2. 그다음, 나머지 9개 직업 카테고리에 대해서도 각각 0~100% 점수를 매기세요.
    3. 마지막으로, 그 9개 직업 카테고리 중 점수가 가장 높은 상위 2개 카테고리 이름과 점수를 추출하세요.
    4. 아래 출력 형식에 맞춰 정확히 결과만 출력하고, 다른 설명이나 해설은 포함하지 마세요.

    출력 형식(예시):
    {jobObjective}: 85%
    서비스업: 78%
    IT·웹·통신: 72%

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

prompt_3 = PromptTemplate(
    input_variables=["resume"],
    template="""
    입력한 자기소개서가 어떤 인재 유형인지 판단해줘.
    책임의식형, 도전정신형, 소통협력형, 창의성형 4가지 중에 하나로 판단해주면 돼.
    책임의식형, 도전정신형, 소통협력형, 창의성형 중 하나로 출력해주면 돼
    
    자기소개서:
    {resume}
    
    인재 유형:
    """
)

prompt_4 = PromptTemplate(
    input_variables=["lorem", "preferred"],
    template="""
    너가 고용주의 입장에서 기업의 인재상에 해당하는 문장과 자기소개서의 직무적인 유사도를 판단해야해.
    0~100% 사이의 점수로 평가하면 돼.
    숫자%만 출력해줘
    그리고 같은 자기소개서와 기업 인재상을 입력하고 실행시키면 매번 다른 출력값이 나오는데 그러지 않게 일관적인 결과를 출력해줘
    
    자기소개서 :
    {lorem}
    
    기업의 인재상 :
    {preferred}
    
    일치도 점수: <숫자>%
    """
)

async def process_pipeline(lorem, jobObjective):
    response_1_obj = await (prompt_1 | model_1).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
    response_1_text = response_1_obj.content
    
    scores = {}
    for line in response_1_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        category, score_part = line.split(":", 1)
        score_str = score_part.strip().replace("%", "")
        try:
            score = int(score_str)
            scores[category.strip()] = score
        except ValueError:
            continue

    job_score = scores.get(jobObjective, 0)
    
    total_score = { jobObjective: job_score }
    
    other_scores = {cat: sc for cat, sc in scores.items() if cat != jobObjective}
    top2 = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    for cat, sc in top2:
        total_score[cat] = sc
        
    if job_score >= 75:
        response_2_obj = await (prompt_2_resume | model_2).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
        response_2 = response_2_obj.content
        return {"ability": response_1_text, "resume": response_2, "lorem": lorem, "total_score": total_score}
    
    else:
        response_2_obj = await (prompt_2_study | model_2).ainvoke({"jobObjective": jobObjective})
        response_2 = response_2_obj.content
        return {"ability": response_1_text, "study": response_2, "lorem": lorem, "total_score": total_score}
    
async def talentedType_pipeline(resume) :
    response_3_obj = await (prompt_3 | model_3).ainvoke({"resume": resume})
    response_3 = response_3_obj.content
    return { "talentedType" : response_3 }

async def similarity_pipeline(lorem, preferred) :
    response_4_obj = await (prompt_4 | model_4).ainvoke({"lorem": lorem, "preferred": preferred})
    response_4 = response_4_obj.content
    return { "similarity" : response_4 }

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
    
class TalentedTypeRequest(BaseModel):
    resume: str
    
class SimilarityRequest(BaseModel):
    lorem: str
    preferred: str

@app.post("/user/validate_resume")
async def validate_resume(request: ResumeRequest):
    print("서버가 정상적으로 연결됐습니다.")
    lorem = request.lorem
    jobObjective = request.jobObjective 
    return await process_pipeline(lorem, jobObjective)

@app.post("/user/talentedType")
async def talentedType(request: TalentedTypeRequest):
    resume = request.resume
    return await talentedType_pipeline(resume)

@app.post("/employer/similarity")
async def similarity(request: SimilarityRequest):
    lorem = request.lorem
    preferred = request.preferred
    return await similarity_pipeline(lorem, preferred)

if __name__ == "__main__":
    uvicorn.run("main:app", host=f"{host}", port=port, reload=True)
