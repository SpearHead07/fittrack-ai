"""
FitTrack API — Day 3
Adds the Claude-powered /coach endpoint.

Run:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from supabase import create_client, Client
from anthropic import AsyncAnthropic
import voyageai
from dotenv import load_dotenv
import os
import json

# ---- Setup ----
load_dotenv()

supabase_anon: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

# Async Claude client — used inside async endpoints so a slow
# 2-3 second LLM call doesn't block other requests.
claude = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Voyage AI for embeddings — Anthropic's recommended partner.
# voyage-3-lite: 512-dim, fast, cheap, asymmetric (doc/query input types).
voyage = voyageai.AsyncClient(api_key=os.environ["VOYAGEAI_API_KEY"])
EMBED_MODEL = "voyage-3-lite"

app = FastAPI(title="FitTrack API", version="0.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- Auth dependency ----
def get_current_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        user = supabase_anon.auth.get_user(token).user
        if user is None:
            raise HTTPException(401, "Invalid token")
        return user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

# ---- Pydantic schemas ----
class MealIn(BaseModel):
    section: Literal["breakfast", "lunch", "dinner", "snacks"]
    name: str
    calories: float = Field(ge=0)
    carbs: float = 0
    protein: float = 0
    fat: float = 0
    log_date: Optional[str] = None

class ExerciseIn(BaseModel):
    name: str
    calories: float = Field(ge=0)
    duration: Optional[int] = None
    log_date: Optional[str] = None

class CoachInput(BaseModel):
    goal: int = Field(ge=500, le=5000)
    eaten: float = Field(ge=0)
    burned: float = Field(ge=0)
    protein: float = 0
    carbs: float = 0
    fat: float = 0
    meal_summary: str = ""

# Day 4: Structured output schema. This is what Claude is FORCED to return.
class CoachAdvice(BaseModel):
    status: str = Field(description="One-line status summary, e.g. 'Under target by 200 kcal'")
    verdict: str = Field(description="2-sentence warm assessment of the user's day")
    tip: str = Field(description="One specific, actionable tip for tomorrow")
    severity: Literal["good", "warn", "bad"] = Field(
        description="'good' if on/under target, 'warn' if slightly over, 'bad' if significantly over"
    )

# ---- Public ----
@app.get("/")
def health():
    return {"status": "ok"}

# ---- Meals ----
@app.post("/meals")
def create_meal(meal: MealIn, user_id: str = Depends(get_current_user)):
    payload = meal.model_dump(exclude_none=True)
    payload["user_id"] = user_id
    return supabase.table("meals").insert(payload).execute().data[0]

@app.get("/meals")
def list_meals(day: Optional[str] = None, user_id: str = Depends(get_current_user)):
    q = supabase.table("meals").select("*").eq("user_id", user_id)
    if day:
        q = q.eq("log_date", day)
    return q.order("created_at").execute().data

@app.delete("/meals/{meal_id}")
def delete_meal(meal_id: str, user_id: str = Depends(get_current_user)):
    supabase.table("meals").delete().eq("id", meal_id).eq("user_id", user_id).execute()
    return {"deleted": meal_id}

# ---- Exercises ----
@app.post("/exercises")
def create_exercise(ex: ExerciseIn, user_id: str = Depends(get_current_user)):
    payload = ex.model_dump(exclude_none=True)
    payload["user_id"] = user_id
    return supabase.table("exercises").insert(payload).execute().data[0]

@app.get("/exercises")
def list_exercises(day: Optional[str] = None, user_id: str = Depends(get_current_user)):
    q = supabase.table("exercises").select("*").eq("user_id", user_id)
    if day:
        q = q.eq("log_date", day)
    return q.order("created_at").execute().data

@app.delete("/exercises/{exercise_id}")
def delete_exercise(exercise_id: str, user_id: str = Depends(get_current_user)):
    supabase.table("exercises").delete().eq("id", exercise_id).eq("user_id", user_id).execute()
    return {"deleted": exercise_id}

# ---- AI Coach (Day 4: structured output via tool use) ----
COACH_SYSTEM_PROMPT = (
    "You are a supportive, evidence-based fitness coach helping the user "
    "lose weight sustainably. Be warm, specific, and never shame or scold. "
    "Do not give medical advice. You MUST respond by calling the "
    "provide_coaching tool."
)

# Tool schema — derived from the Pydantic model so the two stay in sync.
COACH_TOOL = {
    "name": "provide_coaching",
    "description": "Return structured coaching feedback for the user's daily log.",
    "input_schema": CoachAdvice.model_json_schema(),
}

@app.post("/coach")
async def coach(payload: CoachInput, user_id: str = Depends(get_current_user)):
    net = payload.eaten - payload.burned
    diff = net - payload.goal

    user_message = (
        f"Today's data:\n"
        f"- Target intake: {payload.goal} kcal\n"
        f"- Eaten: {payload.eaten:.0f} kcal\n"
        f"- Burned via exercise: {payload.burned:.0f} kcal\n"
        f"- Net intake: {net:.0f} kcal\n"
        f"- Difference from target: {diff:+.0f} kcal\n"
        f"- Protein: {payload.protein:.0f}g, Carbs: {payload.carbs:.0f}g, Fat: {payload.fat:.0f}g\n"
        f"- Meals logged: {payload.meal_summary or 'none'}\n\n"
        f"Call the provide_coaching tool with your assessment."
    )

    try:
        response = await claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            system=COACH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[COACH_TOOL],
            tool_choice={"type": "tool", "name": "provide_coaching"},
        )
    except Exception as e:
        raise HTTPException(503, f"Coach unavailable: {e}")

    # Find the tool_use block in the response
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise HTTPException(500, "Model did not call the provide_coaching tool")

    # Defense in depth: validate the model's structured output against Pydantic
    try:
        advice = CoachAdvice.model_validate(tool_block.input)
    except Exception as e:
        raise HTTPException(500, f"Invalid model output: {e}")

    return {
        "advice": advice.model_dump(),
        "usage": response.usage.model_dump(),
        "model": response.model,
    }

# ---- Day 5: Semantic food search (RAG retrieval) ----
@app.get("/search-food")
async def search_food(q: str, limit: int = 8, user_id: str = Depends(get_current_user)):
    """
    Embed the user's query with Voyage (input_type='query'), then run
    a vector-similarity search against the foods table via the
    match_foods Postgres RPC.

    Note: input_type='query' is critical — Voyage tunes the embedding
    for the role of the text. Using 'document' here would still work
    but quality would drop.
    """
    if not q or len(q.strip()) < 2:
        return []

    try:
        emb_resp = await voyage.embed(
            texts=[q.strip()],
            model=EMBED_MODEL,
            input_type="query",
        )
    except Exception as e:
        raise HTTPException(503, f"Embedding service unavailable: {e}")

    query_embedding = emb_resp.embeddings[0]

    result = supabase.rpc("match_foods", {
        "query_embedding": query_embedding,
        "match_count": limit,
    }).execute()

    return result.data


# =============================================================
# Day 6: Agent loop — natural-language meal/exercise logging
# =============================================================

AGENT_SYSTEM_PROMPT = """
You are FitTrack's meal logging assistant. The user describes what they ate or did
in natural language. Your job: log everything accurately by calling tools.

Workflow:
1. Identify each food and each exercise mentioned in the input.
2. For each food, call search_food first to find accurate nutrition values.
3. Call log_meal for each food. Infer the section (breakfast/lunch/dinner/snacks)
   from time-of-day clues; default to 'snacks' if unclear.
4. Call log_exercise for activities; estimate calories from duration if needed.
5. After all tools succeed, respond with a brief plain-text summary.

Rules:
- If user says "two rotis", multiply per-unit calories by 2 (or call log_meal twice).
- Always search before logging. Never invent nutrition values.
- You may call multiple tools in a single turn (parallel) when independent.
- Keep the final summary under 30 words.
""".strip()

AGENT_TOOLS = [
    {
        "name": "search_food",
        "description": (
            "Search the foods database semantically. Call this BEFORE log_meal to "
            "find accurate nutrition values for foods the user mentioned. Returns "
            "top matches with calories, carbs, protein, fat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Food name, e.g. 'roti', 'chicken curry'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "log_meal",
        "description": (
            "Log one meal entry. If user said 'two X', scale calories/macros accordingly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snacks"]},
                "name": {"type": "string"},
                "calories": {"type": "number"},
                "carbs": {"type": "number"},
                "protein": {"type": "number"},
                "fat": {"type": "number"},
            },
            "required": ["section", "name", "calories"],
        },
    },
    {
        "name": "log_exercise",
        "description": (
            "Log an exercise. Estimate calories from duration: "
            "running ~10 kcal/min, walking ~4 kcal/min, cycling ~8 kcal/min, "
            "yoga ~5 kcal/min, weight training ~7 kcal/min."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "calories": {"type": "number"},
                "duration": {"type": "integer", "description": "minutes"},
            },
            "required": ["name", "calories"],
        },
    },
]


async def execute_tool(name: str, args: dict, user_id: str):
    """Dispatch a tool_use block to the right backend logic."""
    if name == "search_food":
        emb = await voyage.embed(
            texts=[args["query"]],
            model=EMBED_MODEL,
            input_type="query",
        )
        result = supabase.rpc("match_foods", {
            "query_embedding": emb.embeddings[0],
            "match_count": 3,
        }).execute()
        return result.data

    if name == "log_meal":
        payload = {**args, "user_id": user_id}
        return supabase.table("meals").insert(payload).execute().data[0]

    if name == "log_exercise":
        payload = {**args, "user_id": user_id}
        return supabase.table("exercises").insert(payload).execute().data[0]

    raise ValueError(f"Unknown tool: {name}")


class AgentInput(BaseModel):
    text: str = Field(min_length=2, max_length=500)


@app.post("/quick-log")
async def quick_log(payload: AgentInput, user_id: str = Depends(get_current_user)):
    """
    The agent endpoint. Takes one natural-language line, runs the loop
    (Claude -> tool calls -> results -> Claude ...) until end_turn,
    returns a summary plus the list of actions taken.
    """
    messages = [{"role": "user", "content": payload.text}]
    actions = []

    for step in range(8):                                # safety cap
        response = await claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        # Remember Claude's turn in the conversation history
        messages.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        })

        if response.stop_reason == "end_turn":
            final = next((b.text for b in response.content if b.type == "text"), "Done.")
            return {"summary": final, "actions": actions, "steps": step + 1}

        if response.stop_reason != "tool_use":
            raise HTTPException(500, f"Unexpected stop_reason: {response.stop_reason}")

        # Execute every tool_use block from this turn
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = await execute_tool(block.name, block.input, user_id)
                is_error = False
            except Exception as e:
                result = {"error": str(e)}
                is_error = True
            actions.append({"tool": block.name, "input": block.input, "output": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
                "is_error": is_error,
            })

        # Feed all tool results back to Claude as one user message
        messages.append({"role": "user", "content": tool_results})

    raise HTTPException(500, "Agent exceeded max iterations")