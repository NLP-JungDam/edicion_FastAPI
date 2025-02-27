import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수에 OPENAI_API_KEY를 설정하세요.")
os.environ["OPENAI_API_KEY"] = api_key

host = os.getenv("SERVER_HOST")
port = int(os.getenv("SERVER_PORT"))

embedding_ada002 = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
embedding_3small = OpenAIEmbeddings(model="text-embedding-3-small")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=43)

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "ability.txt")

loader = TextLoader(file_path, encoding="utf-8")

split_doc = loader.load_and_split(text_splitter)

splits = text_splitter.split_documents(split_doc)

competency_db = Chroma.from_documents(
    documents=split_doc, embedding=embedding_ada002, collection_name="competency", persist_directory='./data/'
)

ability_db = Chroma(persist_directory="./data/", embedding_function=embedding_3small)

model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")

# 잡코리아, 사람인, 한국 산업인력 공단에서 가져온 직무별 핵심 역량 키워드
competency_dataset = [
    {
        "category": "서비스업",
        "core_competencies": [
            "고객 응대", "커뮤니케이션", "문제 해결", "친절함", "스트레스 관리",
            "서비스 마인드", "감정 조절", "고객 니즈 파악", "상황 대처 능력"
        ],
        "keywords": ["고객 만족", "서비스 마인드", "상담", "대인 관계", "응대 기술"],
        "description": "고객과 직접 대면하며 만족도를 높이기 위한 대인 관계 능력과 문제 해결, 감정 조절 및 고객 니즈 파악이 중요한 분야."
    },
    {
        "category": "제조·화학",
        "core_competencies": [
            "생산 공정 관리", "품질 관리", "안전 규정 준수", "기술적 이해", "문제 진단",
            "설비 운영", "공정 개선", "원가 절감", "환경 관리", "데이터 분석"
        ],
        "keywords": ["생산 효율", "품질 보증", "공정 개선", "안전 관리", "화학 지식"],
        "description": "효율적인 생산과 품질 유지, 안전 및 환경 관리를 통한 공정의 최적화와 원가 절감 능력이 요구되는 분야."
    },
    {
        "category": "IT·웹·통신",
        "core_competencies": [
            "프로그래밍", "시스템 설계", "네트워크 관리", "문제 해결", "최신 기술 동향 파악",
            "데이터 분석", "클라우드 컴퓨팅", "소프트웨어 아키텍처", "보안 인식", "협업 능력", "AI"
        ],
        "keywords": ["코딩", "소프트웨어 개발", "데이터베이스", "클라우드", "알고리즘", "보안"],
        "description": "다양한 프로그래밍 언어와 기술 스택을 활용하여 시스템을 구축·운영하며, 문제 해결 및 협업, 최신 기술 동향 파악 능력이 중요한 분야."
    },
    {
        "category": "은행·금융업",
        "core_competencies": [
            "금융 분석", "리스크 관리", "데이터 분석", "고객 상담", "규제 준수",
            "재무 모델링", "투자 평가", "시장 조사", "의사소통", "윤리 경영"
        ],
        "keywords": ["재무 분석", "투자 전략", "금융 상품", "신용 평가", "자산 관리"],
        "description": "정량적 분석과 고객 관리, 규제 준수를 통한 안정적인 금융 운영과 재무 모델링, 시장 조사 능력이 요구되는 분야."
    },
    {
        "category": "미디어·디자인",
        "core_competencies": [
            "창의력", "디자인 툴 활용", "콘텐츠 기획", "트렌드 분석", "커뮤니케이션",
            "스토리텔링", "브랜드 전략", "비주얼 커뮤니케이션", "프로젝트 관리", "시장 조사"
        ],
        "keywords": ["그래픽 디자인", "UX/UI", "영상 편집", "브랜딩", "크리에이티브"],
        "description": "시각적 표현과 콘텐츠 기획, 최신 트렌드와 브랜드 전략을 통한 창의적 디자인 역량 및 스토리텔링 능력이 핵심인 분야."
    },
    {
        "category": "교육업",
        "core_competencies": [
            "교육 기획", "강의 능력", "커뮤니케이션", "멘토링", "학습자 분석",
            "교육 평가", "문제 해결", "창의적 교수법", "디지털 교육 도구 활용", "조직 운영"
        ],
        "keywords": ["교수법", "커리큘럼 개발", "학습 동기 부여", "평가", "피드백"],
        "description": "효과적인 교육 프로그램 기획과 강의, 학습자 개개인의 이해도를 분석 및 평가하고, 디지털 교육 도구 활용 능력이 중요한 분야."
    },
    {
        "category": "의료·제약·복지",
        "core_competencies": [
            "전문 지식", "환자 중심 케어", "윤리 의식", "연구 및 분석", "팀워크",
            "임상 판단", "문제 해결", "커뮤니케이션", "데이터 해석", "위기 관리"
        ],
        "keywords": ["진단", "치료 계획", "약물 관리", "의료 기술", "복지 서비스"],
        "description": "정밀한 전문 지식을 바탕으로 환자 케어 및 연구, 윤리적 책임과 팀워크, 임상 판단 및 위기 관리 능력이 요구되는 분야."
    },
    {
        "category": "판매·유통",
        "core_competencies": [
            "영업 전략", "고객 관리", "협상력", "시장 분석", "재고 관리",
            "마케팅 전략", "CRM 활용", "커뮤니케이션", "네트워킹"
        ],
        "keywords": ["판매 목표", "마케팅", "고객 확보", "CRM", "유통 채널"],
        "description": "효과적인 영업 전략과 고객 관리, 데이터 분석 및 마케팅 전략 수립을 통해 판매 및 유통 채널을 최적화하는 능력이 중요한 분야."
    },
    {
        "category": "건설업",
        "core_competencies": [
            "프로젝트 관리", "현장 운영", "안전 관리", "기술적 전문성", "협업",
            "계획 수립", "문제 해결", "비용 관리", "리더십", "현장 소통"
        ],
        "keywords": ["공사 관리", "건축 설계", "엔지니어링", "현장 안전", "시공 능력"],
        "description": "프로젝트 관리와 현장 운영, 기술적 전문성, 비용 관리 및 현장 소통을 통한 안전하고 효율적인 건설 수행 능력이 요구되는 분야."
    },
    {
        "category": "기관·협회",
        "core_competencies": [
            "정책 이해", "조직 관리", "커뮤니케이션", "네트워킹", "문서 작성",
            "리더십", "분석적 사고", "전략 기획", "협상력", "대외 협력"
        ],
        "keywords": ["행정", "법규 준수", "공공 서비스", "협력", "리더십"],
        "description": "공공기관이나 협회에서 정책 이해와 조직 관리, 효과적인 커뮤니케이션 및 네트워킹을 통한 전략 기획과 대외 협력 능력이 요구되는 분야."
    }
]

def build_category_text(entry):
    return (
        f"카테고리: {entry['category']}\n"
        f"핵심 역량: {', '.join(entry['core_competencies'])}\n"
        f"키워드: {', '.join(entry['keywords'])}\n"
        f"설명: {entry['description']}"
    )
    
category_texts = [build_category_text(entry) for entry in competency_dataset]
    
def compute_similarity_scores(lorem_text):
    lorem_embedding = embedding_ada002.embed_query(lorem_text)
    scores = {}
    for entry, cat_text in zip(competency_dataset, category_texts):
        cat_embedding = embedding_ada002.embed_query(cat_text)
        sim = cosine_similarity([lorem_embedding], [cat_embedding])[0][0]
        # 코사인 유사도는 [0,1] 범위로 나오므로 100을 곱해 정수 점수로 변환
        score = int(sim * 100)
        scores[entry["category"]] = score
    return scores

base_prompt = PromptTemplate(
    input_variables=["lorem"],
    template="""
    사용자가 입력한 텍스트가 자기소개 형태인지를 판단하는 AI야
    자기를 소개하는 글이라고 판단되는 확률을 0~100%사이로 출력해줘
    숫자%로만 출력해줘
    
    사용자가 입력한 자기소개 :
    {lorem}
    
    자기소개 형태 판단:<숫자>%
    """
)

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
    input_variables=["jobObjective", "retrieved_text"],
    template="""
    사용자의 자기소개서가 {jobObjective}에 적합하지 않음.
    따라서 {jobObjective}에 맞는 실력을 키우기 위한 공부법과 방향을 제공해줘.
    
    참고할 내용:
    {retrieved_text}

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
    input_variables=["lorem", "preferred", "job_id"],
    template="""
    너가 고용주의 입장에서 {job_id}기업의 인재상에 해당하는 문장과 자기소개서의 직무적인 유사도를 판단해야해.
    0~100% 사이의 점수로 평가하면 돼.
    숫자%만 출력해줘
    
    자기소개서 :
    {lorem}
    
    기업의 인재상 :
    {preferred}
    
    일치도 점수: <숫자>%
    """
)

async def resume_pipeline(lorem, jobObjective):
    checkResponse = await (base_prompt | model).ainvoke({"lorem" : lorem})
    content = checkResponse.content.strip().replace("%", "")
    checkResponse = int(content)
    if checkResponse <= 80  :
        return { "verify" : False }
    
    vector_scores = compute_similarity_scores(lorem)
    
    response_1_obj = await (prompt_1 | model).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
    response_1_text = response_1_obj.content
    
    # jobObjective에 해당하는 점수는 벡터 점수를 우선 사용하도록 예시 처리합니다.
        
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

    vector_score = vector_scores.get(jobObjective, 0)
    job_score = scores.get(jobObjective, 0)
    
    total_score = { jobObjective: (job_score+vector_score)//2 }
    
    other_scores = {cat: sc for cat, sc in scores.items() if cat != jobObjective}
    top2 = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    for cat, sc in top2:
        total_score[cat] = sc
        
    if job_score >= 75:
        response_2_obj = await (prompt_2_resume | model).ainvoke({"lorem": lorem, "jobObjective": jobObjective})
        response_2 = response_2_obj.content
        
        return {"ability": response_1_text, "resume": response_2, "lorem": lorem, "total_score": total_score}
    
    else:
        retrieved_text = competency_db.similarity_search(f"{response_1_text}에 대한 역량을 키우기 위한 방법", k=1)
        retrieved_text = retrieved_text[0].page_content
        response_2_obj = await (prompt_2_study | model).ainvoke({"jobObjective": jobObjective, "retrieved_text": retrieved_text})
        response_2 = response_2_obj.content
        return {"ability": response_1_text, "study": response_2, "lorem": lorem, "total_score": total_score}
    
async def talentedType_pipeline(resume) :
    response_3_obj = await (prompt_3 | model).ainvoke({"resume": resume})
    response_3 = response_3_obj.content
    return { "talentedType" : response_3 }

async def similarity_pipeline(lorem, jobs):
    gpt_scores = {}

    for job_id, preferred in jobs.items():
        response_4_obj = await (prompt_4 | model).ainvoke({"lorem": lorem, "preferred": preferred,"job_id":job_id})
        gpt_scores[f"{job_id}"] = float(response_4_obj.content.replace("%", ""))

    return gpt_scores

async def calculate_cosine_similarity(lorem, jobs):
    lorem_embedding = embedding_3small.embed_query(lorem)
    
    similarity_scores = {}
    
    for job_id, preferred in jobs.items():
        preferred_embedding = embedding_3small.embed_query(preferred)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([lorem_embedding], [preferred_embedding])[0][0]
        similarity_scores[job_id] = round(similarity * 100, 2)  # 확률(%) 변환
    
    return similarity_scores

app = FastAPI()

# CORS 미들웨어 추가 (외부 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeRequest(BaseModel):
    lorem: str
    jobObjective: str
    
class TalentedTypeRequest(BaseModel):
    resume: str
    
class SimilarityRequest(BaseModel):
    lorem: str
    jobs: dict
    
class PreferrredRequest(BaseModel):
    businessNumber: str
    preferred: str

@app.post("/user/validate_resume")
async def validate_resume(request: ResumeRequest):
    lorem = request.lorem
    jobObjective = request.jobObjective 
    return await resume_pipeline(lorem, jobObjective)

@app.post("/user/talentedType")
async def talentedType(request: TalentedTypeRequest):
    resume = request.resume
    return await talentedType_pipeline(resume)

@app.post("/employer/similarity")
async def similarity(request: SimilarityRequest):
    lorem = request.lorem
    jobs = request.jobs
    chroma_scores = await calculate_cosine_similarity(lorem, jobs)
    gpt_scores = await similarity_pipeline(lorem, jobs)
    
    final_scores = {
        job_id: round((gpt_scores[job_id] + chroma_scores[job_id]) / 2, 2)
        for job_id in request.jobs.keys()
    }
    
    return {"chroma_scores": chroma_scores, "gpt_scores": gpt_scores, "fitness": final_scores}

if __name__ == "__main__":
    uvicorn.run("main:app", host=f"{host}", port=port, reload=True)