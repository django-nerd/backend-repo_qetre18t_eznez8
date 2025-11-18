import os
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document, get_documents
from schemas import Project, Prompt, Run, Template
import requests

app = FastAPI(title="PromptForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Utility: heuristic scoring
class HeuristicScore(BaseModel):
    score: float
    reasons: List[str]


def heuristic_score(prompt_text: str) -> HeuristicScore:
    reasons: List[str] = []
    score = 0.5
    if len(prompt_text) > 100:
        score += 0.1
        reasons.append("Sufficient length for clarity")
    if any(h in prompt_text.lower() for h in ["role:", "you are", "assistant:"]):
        score += 0.1
        reasons.append("Has role framing")
    if any(h in prompt_text.lower() for h in ["steps:", "bullets:", "format:", "output as"]):
        score += 0.1
        reasons.append("Specifies format/structure")
    if any(h in prompt_text.lower() for h in ["constraints", "rules", "avoid"]):
        score += 0.1
        reasons.append("Includes constraints or guardrails")
    if "example" in prompt_text.lower():
        score += 0.1
        reasons.append("Includes examples")
    return HeuristicScore(score=max(0.0, min(1.0, score)), reasons=reasons)


# Health & schema
@app.get("/")
def root():
    return {"message": "PromptForge API running"}


@app.get("/schema")
def get_schema():
    # Expose model field schemas for viewer tools
    return {
        "project": Project.model_json_schema(),
        "prompt": Prompt.model_json_schema(),
        "run": Run.model_json_schema(),
        "template": Template.model_json_schema(),
    }


# Projects
@app.post("/projects")
def create_project(project: Project):
    project_id = create_document("project", project)
    return {"id": project_id}


@app.get("/projects")
def list_projects() -> List[Dict[str, Any]]:
    return get_documents("project")


# Prompts
class CreatePromptRequest(Prompt):
    pass


@app.post("/prompts")
def create_prompt(payload: CreatePromptRequest):
    # Build optimized prompt using a simple template for now
    template = (
        "Role: {audience}\n"
        "Goal: {instructions}\n\n"
        "Context:\n{context}\n\n"
        "Constraints:\n{constraints}\n\n"
        "Output Format:\n{format}\n\n"
        "Examples:\n{examples}\n"
    )

    optimized = template.format(
        audience=payload.audience or "You are an expert assistant.",
        instructions=payload.instructions,
        context=payload.context or "",
        constraints=payload.constraints or "",
        format=payload.format or "",
        examples=payload.examples or "",
    ).strip()

    hs = heuristic_score(optimized)

    to_save = payload.model_dump()
    to_save.update({
        "optimized_prompt": optimized,
        "score": hs.score,
    })

    prompt_id = create_document("prompt", to_save)
    return {"id": prompt_id, "optimized_prompt": optimized, "score": hs.score, "reasons": hs.reasons}


@app.get("/prompts")
def list_prompts(project_id: Optional[str] = None):
    filt = {"project_id": project_id} if project_id else {}
    return get_documents("prompt", filt)


# Runs: test a prompt with OpenRouter (if key present), otherwise echo
class RunRequest(BaseModel):
    prompt_id: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 512


@app.post("/runs/test")
def run_test(req: RunRequest):
    # Fetch prompt
    docs = get_documents("prompt", {"_id": {"$exists": True}})
    target = None
    for d in docs:
        if str(d.get("_id")) == req.prompt_id:
            target = d
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    input_prompt = target.get("optimized_prompt") or target.get("instructions")
    start = time.time()
    output = None
    cost = None

    if OPENROUTER_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": req.model,
                "messages": [
                    {"role": "user", "content": input_prompt},
                ],
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            }
            resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            cost = data.get("usage", {}).get("total_cost", None)
        except Exception as e:
            output = f"[OpenRouter error: {e}]\n\nEcho:\n{input_prompt}"
    else:
        output = f"[Mock Output]\n\n{input_prompt}"

    latency_ms = int((time.time() - start) * 1000)
    hs = heuristic_score(output)

    run_doc = Run(
        prompt_id=req.prompt_id,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        input_prompt=input_prompt,
        output=output,
        latency_ms=latency_ms,
        cost_usd=cost,
        score=hs.score,
        meta={"reasons": hs.reasons},
    )
    run_id = create_document("run", run_doc)
    return {"id": run_id, "output": output, "latency_ms": latency_ms, "score": hs.score}


@app.get("/runs")
def list_runs(prompt_id: Optional[str] = None):
    filt = {"prompt_id": prompt_id} if prompt_id else {}
    return get_documents("run", filt)


# Templates (basic CRUD)
@app.post("/templates")
def create_template(tpl: Template):
    tpl_id = create_document("template", tpl)
    return {"id": tpl_id}


@app.get("/templates")
def list_templates():
    return get_documents("template")


# Compatibility endpoints from scaffold
@app.get("/api/hello")
def hello():
    return {"message": "Hello from PromptForge backend!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
