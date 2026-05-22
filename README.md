# FitTrack API

A FastAPI backend for a calorie and fitness tracking app powered by Claude AI and Supabase.

## Features

- **Meal logging** — log meals by section (breakfast, lunch, dinner, snacks) with calories and macros
- **Exercise logging** — track workouts with duration and calories burned
- **AI Coach** — daily nutrition feedback powered by Claude (structured output via tool use)
- **Semantic food search** — vector search over a food database using Voyage AI embeddings
- **Natural-language quick-log** — describe what you ate in plain English; the agent searches, parses, and logs everything automatically

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Database & auth | Supabase (Postgres + Row Level Security) |
| AI model | Claude Haiku 4.5 (via Anthropic SDK) |
| Embeddings | Voyage AI `voyage-3-lite` (512-dim) |
| Validation | Pydantic v2 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/meals` | Log a meal |
| `GET` | `/meals` | List meals (filter by `?day=YYYY-MM-DD`) |
| `DELETE` | `/meals/{id}` | Delete a meal |
| `POST` | `/exercises` | Log an exercise |
| `GET` | `/exercises` | List exercises (filter by `?day=YYYY-MM-DD`) |
| `DELETE` | `/exercises/{id}` | Delete an exercise |
| `POST` | `/coach` | Get AI coaching feedback for the day |
| `GET` | `/search-food` | Semantic search over food database (`?q=query`) |
| `POST` | `/quick-log` | Natural-language meal/exercise logging agent |

All endpoints except `GET /` require a Supabase JWT in the `Authorization: Bearer <token>` header.

## Getting Started

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project with `meals`, `exercises`, and `foods` tables
- An [Anthropic](https://console.anthropic.com) API key
- A [Voyage AI](https://www.voyageai.com) API key

### Installation

```bash
cd fittrack-api
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in `fittrack-api/`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
ANTHROPIC_API_KEY=sk-ant-...
VOYAGEAI_API_KEY=pa-...
```

### Seed the Food Database

Run once after setting up the `foods` table in Supabase:

```bash
python seed_foods.py
```

This embeds ~45 common foods (Indian and Western) using Voyage AI and inserts them into the `foods` table. Re-running is idempotent — it wipes and re-seeds.

### Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Supabase Schema

You need the following tables and a vector-search RPC:

```sql
-- Meals
create table meals (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  section text not null check (section in ('breakfast','lunch','dinner','snacks')),
  name text not null,
  calories float not null,
  carbs float default 0,
  protein float default 0,
  fat float default 0,
  log_date date default current_date,
  created_at timestamptz default now()
);

-- Exercises
create table exercises (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  name text not null,
  calories float not null,
  duration int,
  log_date date default current_date,
  created_at timestamptz default now()
);

-- Foods (enable pgvector extension first)
create extension if not exists vector;

create table foods (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  calories float,
  carbs float,
  protein float,
  fat float,
  description text,
  embedding vector(512)
);

-- Vector similarity RPC
create or replace function match_foods(query_embedding vector(512), match_count int)
returns table (id uuid, name text, calories float, carbs float, protein float, fat float, similarity float)
language sql stable as $$
  select id, name, calories, carbs, protein, fat,
         1 - (embedding <=> query_embedding) as similarity
  from foods
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

Enable Row Level Security on `meals` and `exercises` and add policies so users can only access their own rows.

## How the AI Features Work

### `/coach` — Structured AI Feedback

Sends the day's calorie and macro summary to Claude Haiku. Claude is forced (via `tool_choice`) to call a `provide_coaching` tool, which maps directly to a Pydantic model. This guarantees the response is always valid JSON with `status`, `verdict`, `tip`, and `severity` fields.

### `/search-food` — Semantic Search (RAG)

The user's query is embedded with Voyage AI using `input_type="query"`, then matched against pre-embedded food descriptions stored in Supabase using cosine similarity (`pgvector`).

### `/quick-log` — Agent Loop

An agentic Claude loop that:
1. Receives a natural-language description (e.g. "had two rotis and chai for breakfast")
2. Calls `search_food` to look up accurate nutrition values
3. Calls `log_meal` or `log_exercise` for each item found
4. Returns a plain-text summary of what was logged

The loop runs for a maximum of 8 steps to prevent runaway iterations.
