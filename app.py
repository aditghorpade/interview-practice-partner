from pathlib import Path
import os
import json

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# ============================================================
# Environment & Gemini setup
# ============================================================

# Expect a .env file next to this script with GOOGLE_API_KEY
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

print("DEBUG LOADED:", ENV_PATH.exists(), os.getenv("GOOGLE_API_KEY") is not None)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-flash-latest")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner=False)
def get_gemini_model():
    """Return a cached Gemini GenerativeModel instance."""
    return genai.GenerativeModel(MODEL_NAME)


model = get_gemini_model()

# ============================================================
# Streamlit app configuration
# ============================================================

st.set_page_config(page_title="Interview Practice Partner", page_icon="🎤")

st.title("🎤 Interview Practice Partner (Gemini Flash)")
st.write(
    "Practice job interviews with an intelligent AI interviewer powered by "
    "**Gemini Flash** over the internet."
)

# ============================================================
# Session state initialization
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = "setup"

if "role" not in st.session_state:
    st.session_state.role = None

if "mode" not in st.session_state:
    st.session_state.mode = None

# last inferred persona label for the candidate
if "persona" not in st.session_state:
    st.session_state.persona = "Balanced"

if "num_questions" not in st.session_state:
    st.session_state.num_questions = 5

if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

if "qa_list" not in st.session_state:
    st.session_state.qa_list = []


def reset_interview():
    """Reset the whole interview state so the user can start fresh."""
    st.session_state.stage = "setup"
    st.session_state.current_question_index = 0
    st.session_state.qa_list = []
    st.session_state.persona = "Balanced"

    # Clear per-question cache
    for k in list(st.session_state.keys()):
        if k.startswith("question_") or k.startswith("analysis_"):
            del st.session_state[k]


# ============================================================
# Helper: safe JSON parsing
# ============================================================

def try_parse_json(text: str):
    """
    Try to parse a string as JSON.

    Handles common patterns like:
    - ```json ... ```
    - ``` ... ```
    and returns None if parsing fails.
    """
    if not text:
        return None

    s = text.strip()

    # Strip off markdown code fences if present
    if s.startswith("```"):
        s = s.strip("`")
        # Sometimes the first line is 'json'
        if s.lower().startswith("json"):
            s = s[s.find("\n") + 1:].strip()

    try:
        return json.loads(s)
    except Exception:
        return None


# ============================================================
# Persona inference based on answers
# ============================================================

def infer_persona(qa_list):
    """
    Infer a rough communication style/persona from the latest meaningful answer.

    Priority:
      1. Confused / Unsure
      2. Very Efficient (short but solid answers)
      3. Technical / Detail-Oriented
      4. Chatty (only if long + confident)
      5. Balanced (fallback)
    """
    if not qa_list:
        return "Balanced"

    last_answer = None
    last_issue = "ok"

    # Find the last non-empty, non-skipped answer
    for item in reversed(qa_list):
        a = (item.get("a") or "").strip()
        if a and a not in ("[No answer]", "[Skipped]"):
            last_answer = a.lower()
            last_issue = item.get("issue", "ok")
            break

    if not last_answer:
        return "Balanced"

    ans = last_answer
    word_count = len(ans.split())
    sentence_count = ans.count(".") + ans.count("!") + ans.count("?") + ans.count("\n")

    # 1) Confused / unsure markers (highest priority)
    confused_markers = [
        "not really sure",
        "i'm not sure",
        "not sure",
        "i guess",
        "i think?",
        "i don't know",
        "dont know",
        "no idea",
        "still figuring out",
        "confused",
        "maybe",
        "not confident",
        "i am lost",
        "i'm lost",
    ]
    if any(p in ans for p in confused_markers) or last_issue in ("too_short", "incomplete"):
        return "Confused / Unsure"

    # 2) Very efficient: short but solid answers
    if 5 <= word_count <= 40 and last_issue == "ok":
        return "Very Efficient (short, focused answers)"

    # 3) Technical / detail-oriented
    tech_keywords = [
        "c", "c++", "java", "python", "django", "react", "api", "sql", "database",
        "docker", "time complexity", "multithreading", "multi-threading",
        "memory", "pipeline", "architecture",
    ]
    if any(t in ans for t in tech_keywords):
        return "Technical / Detail-Oriented"

    # 4) Chatty — only if long and confident
    if word_count > 80 or sentence_count >= 4:
        confident_markers = [
            "i worked on",
            "i built",
            "i developed",
            "i implemented",
            "i designed",
        ]
        if any(p in ans for p in confident_markers):
            return "Chatty (talks a lot, easily off-topic)"

    # 5) Default
    return "Balanced"


# ============================================================
# AI: Answer analysis
# ============================================================

def analyze_answer(question, answer, role, mode):
    """
    Use Gemini to classify the quality of an answer and generate a short coaching tip.

    Returns: (label, coach_message)

    label ∈ { "ok", "off_topic", "too_short", "too_vague", "incorrect", "incomplete" }
    """
    if not answer or answer.strip() == "":
        return (
            "too_short",
            "Your answer is empty or very short. Try to give at least 3–4 sentences with a concrete example.",
        )

    prompt = f"""
You are evaluating an interview answer.

Role: {role}
Interview type: {mode}

Question: {question}
Answer: {answer}

Classify the answer and give a short coaching tip.

Rules:
- Think carefully but do NOT show your reasoning.
- Decide ONE main label from this set:
  "ok", "off_topic", "too_short", "too_vague", "incorrect", "incomplete"
- Then write a very short (1–2 sentence) coaching message for the candidate.

Respond ONLY as JSON, no extra text, in this format:
{{
  "label": "...",
  "coach_message": "..."
}}
"""

    try:
        resp = model.generate_content(prompt)
        parsed = try_parse_json(resp.text)
        if parsed and "label" in parsed and "coach_message" in parsed:
            return parsed["label"], parsed["coach_message"]
    except Exception as e:
        # If the API fails, don't block the flow
        return "ok", f"(Could not analyze answer due to an error: {e})"

    # If parsing fails, just accept the answer
    return "ok", "Answer accepted. You can still improve by giving more structure and specific examples."


# ============================================================
# AI: Live note from intro answer (hidden to user)
# ============================================================

def generate_live_note(intro_answer, role):
    prompt = f"""
Extract a very short structured summary from this interview introduction.

Answer: {intro_answer}

Summarize in this JSON format:

{{
  "background": "...",
  "skills": ["...", "..."],
  "experience_level": "...",
  "personality": "...",
  "confidence": "low|medium|high"
}}

Rules:
- Keep skills broad (e.g., "web development", "Java", "problem solving")
- Infer personality traits if possible (e.g., shy, confident, concise, talkative)
- Infer experience level (e.g., beginner, intermediate, experienced)
- MAX 2 sentences worth of content
"""

    try:
        resp = model.generate_content(prompt)
        data = try_parse_json(resp.text)
        return data or {}
    except Exception:
        return {}


# ============================================================
# AI: Example answer generator
# ============================================================

def generate_example_answer(question, role):
    prompt = f"""
You are helping a candidate prepare for a job interview.

Role: {role}
Question: {question}

Write a strong but concise sample answer (4–6 sentences) that:
- Clearly answers the question
- Shows relevant skills and experience
- Sounds like a realistic junior/intermediate candidate (not a superhero)

Respond with ONLY the answer text, no explanation or bullet points.
"""
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        # Simple hard-coded fallback if the API fails
        return (
            "Hi, I am Rajan Mahajan, a Computer Engineering graduate with a strong interest in building reliable, "
            "scalable software. During college I worked on a multithreaded C project that taught me how to reason "
            "about performance and concurrency. I enjoy turning messy problems into clear, well-structured code, and "
            "I'm excited about this Software Engineer role because it would let me grow my skills while contributing "
            "to real products that impact users."
        )


# ============================================================
# AI: Adaptive question generation
# ============================================================

def generate_ai_question(role, mode, persona, previous_qa):
    """
    Generate the next interview question.

    Behavior:
    - Role-aware (Software Engineer / Sales Associate / Data Analyst / Product Manager)
    - Adjusts based on:
        * candidate persona/style
        * previous answers
        * specific keywords (metrics, tech, scenarios, etc.)
    - Handles:
        * example answer requests
        * repeat requests
        * clarification requests
        * off-topic / policy questions
    """
    import re

    # Ensure session keys exist for cross-question memory
    if "live_note" not in st.session_state:
        st.session_state.live_note = None
    if "asked_topics" not in st.session_state:
        st.session_state.asked_topics = set()
    if "followups_asked" not in st.session_state:
        st.session_state.followups_asked = set()
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "repeat_count" not in st.session_state:
        st.session_state.repeat_count = 0

    def contains(text, patterns):
        t = (text or "").lower()
        return any(p in t for p in patterns)

    def short(s, n=200):
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[: n - 3] + "..."

    # Derive an approximate "stage" of the interview from number of questions
    count = len(previous_qa)
    if count < 3:
        stage = "introductory"
    elif count < 6:
        stage = "exploratory"
    elif count < 9:
        stage = "deep_dive"
    else:
        stage = "synthesis"

    # --------------------------------------------------------
    # First question: a warm intro tailored to the role/persona
    # --------------------------------------------------------
    if not previous_qa:
        tone_map = {
            "confused / unsure": "very gentle, patient, reassuring",
            "very efficient (short, focused answers)": "concise and direct",
            "chatty (talks a lot, easily off-topic)": "friendly but structured",
            "technical / detail-oriented": "precise and professional",
            "balanced": "neutral and conversational",
        }
        tone = tone_map.get((persona or "Balanced"), "neutral and conversational")

        role_intro_map = {
            "software engineer": "introduce yourself and what draws you to software engineering",
            "sales associate": "introduce yourself and what draws you to this sales role",
            "data analyst": "introduce yourself and what draws you to data analysis",
            "product manager": "introduce yourself and what draws you to product management",
        }
        role_lower = (role or "").lower()
        base_intro = role_intro_map.get(
            role_lower,
            f"introduce yourself and tell me why you applied for this {role} role",
        )

        prompt = (
            "You are a friendly interviewer. Produce ONE natural first interview question "
            f"asking the candidate to {base_intro}.\n"
            f"Role: {role}\n"
            f"Tone: {tone}\n"
            "Constraints:\n"
            "- Single sentence\n"
            "- <= 22 words\n"
            "- Do NOT use the word 'prompt'\n"
            "- Sound human and warm\n"
        )
        try:
            resp = model.generate_content(prompt)
            q = resp.text.strip().splitlines()[0].strip()
            if not q.endswith("?"):
                q += "?"
        except Exception:
            q = f"Let's start — could you {base_intro}?"

        st.session_state.last_question = q
        st.session_state.repeat_count = 0
        return q

    # --------------------------------------------------------
    # Build live note after the intro (first real answer)
    # --------------------------------------------------------
    if len(previous_qa) == 1 and st.session_state.live_note is None:
        try:
            st.session_state.live_note = generate_live_note(previous_qa[0]["a"], role)
        except Exception:
            st.session_state.live_note = None

    last = previous_qa[-1]
    last_a = (last.get("a") or "").strip().lower()
    last_issue = last.get("issue", "ok")
    live = st.session_state.live_note or {}
    role_lower = (role or "").lower()

    # --------------------------------------------------------
    # Candidate explicitly asks for an example answer
    # --------------------------------------------------------
    example_phrases = [
        "give an example",
        "give me an example",
        "example answer",
        "sample answer",
        "show me an example",
        "yes please give an example",
        "yes, give an example",
        "can you give an example",
        "how should i answer",
        "show me how to answer",
    ]
    if contains(last_a, example_phrases):
        base_question = st.session_state.last_question or last.get("q") or "that question"
        example = generate_example_answer(base_question, role)

        q = (
            f"Here's a sample way you could answer that:\n\n"
            f"{example}\n\n"
            f"Now, in your own words: {base_question}"
        )

        st.session_state.last_question = base_question
        st.session_state.repeat_count = 0
        return q

    # --------------------------------------------------------
    # Candidate asks to repeat the question
    # --------------------------------------------------------
    repeat_phrases = [
        "repeat",
        "can u repeat",
        "can you repeat",
        "say again",
        "repeat the question",
        "please repeat",
        "i forgot the question",
        "what was the question",
    ]
    if contains(last_a, repeat_phrases):
        if st.session_state.last_question:
            if stage == "introductory":
                rep = (
                    f"Sure — I'll repeat the previous question: {st.session_state.last_question} "
                    "Would you like me to say it more slowly or clarify?"
                )
            else:
                rep = (
                    f"Sure — I'll repeat the previous question: {st.session_state.last_question} "
                    "Would you like me to say it more slowly or give a simple example?"
                )
            return rep
        else:
            return (
                "Sure — which question would you like me to repeat? "
                "(I don't have a previous question stored.)"
            )

    # --------------------------------------------------------
    # Edge case: candidate asks off-topic / policy questions
    # --------------------------------------------------------
    irrelevant_q_patterns = [
        "pet", "pets", "cat", "dog", "animal",
        "tuesday", "weekend", "weekends", "saturday", "sundays", "only on",
        "holiday", "holidays", "vacation", "pto", "leave",
        "work from home", "remote only", "only remote", "hybrid",
        "salary", "pay", "compensation", "stipend",
        "free food", "snacks", "lunch", "benefits", "perks",
        "dress code", "uniform", "clothes",
        "bring my", "can i bring", "do we get", "do i get", "will i get",
        "office environment", "ac", "air conditioning",
        "breaks", "coffee", "parking",
        "visa", "relocation",
    ]

    # Detect candidate asking about off-topic personal/policy matters
    if "?" in last_a and any(p in last_a for p in irrelevant_q_patterns):
        # The part before the first "?" is the user question
        user_q = last_a.split("?")[0].strip()

        # Let the model answer the 1st question in a short, generic, HR-safe way
        try:
            resp = model.generate_content(
                f"""
You are acting as a professional interviewer.

The candidate asked an off-topic question:

"{user_q}"

Provide a SHORT, friendly, HR-safe reply that:
- does NOT imply guarantees
- sounds professional
- acknowledges the question
- gives a reasonable generic policy answer
- avoids legal/contract claims
- 1–2 sentences only
- no bullet points
- no company-specific promises
"""
            )
            policy_answer = (resp.text or "").strip().split("\n")[0]
        except Exception:
            policy_answer = (
                "Good question — policies vary by company, but these things usually depend "
                "on business needs and team guidelines."
            )

        follow = (
            st.session_state.last_question
            or "could you tell me briefly about your previous relevant experience?"
        )

        q = f"{policy_answer}\n\nGetting back on track — {follow}"
        st.session_state.repeat_count = 0
        return q

    # --------------------------------------------------------
    # Candidate asks for clarification / elaboration
    # --------------------------------------------------------
    clarification_phrases = [
        "what do you mean",
        "be more specific",
        "clarify",
        "rephrase",
        "not clear",
        "i didn't understand",
        "can u please elaborate",
        "can you elaborate",
        "please elaborate",
        "elaborate",
        "i am not sure",
        "im not sure",
        "can you explain",
        "could you elaborate",
        "could you explain",
    ]
    if contains(last_a, clarification_phrases) or (
        last_issue in ("too_short", "too_vague") and "elaborate" in last_a
    ):
        # If we have the last stored question, rephrase it simply and provide a short example
        if st.session_state.last_question:
            simple_q = short(st.session_state.last_question, 200)

            example_prompt = f"""
You are an interview coach.

Task:
Generate ONE short example answer (2–3 sentences) that directly answers this interview question:

"{simple_q}"

Role: {role}

Rules:
- Make it realistic for a junior candidate
- Include one concrete detail (project, metric, customer action, etc.)
- No filler text like "here is an example"
- Do NOT include explanations, only the example answer
"""
            try:
                ex_resp = model.generate_content(example_prompt)
                example_text = (ex_resp.text or "").strip().splitlines()[0].strip()
            except Exception:
                example_text = (
                    "I worked on a small project where I improved performance using "
                    "multithreading in C++, which reduced processing time noticeably."
                )

            response = (
                "Of course — let me explain more simply:\n\n"
                f"{simple_q}\n\n"
                f"Example: \"{example_text}\"\n\n"
                "Does that help? Would you like another example or would you like to answer now?"
            )

            st.session_state.last_question = simple_q
            return response

    # --------------------------------------------------------
    # Early role-specific context anchoring
    # --------------------------------------------------------
    text_all = " ".join(x.get("a", "") for x in previous_qa)

    # Patterns to detect partial context for different roles
    se_anchor_pattern = re.compile(
        r"\b(scale|scalability|scalable|high concurrency|throughput|latency|architecture|distributed|system design)\b",
        flags=re.I,
    )
    se_context_pattern = re.compile(
        r"\b(project|service|module|microservice|system|deployment|production|internship|company)\b",
        flags=re.I,
    )

    sales_anchor_pattern = re.compile(
        r"\b(sales|targets?|quota|closing|upsell|cross-sell|crm|pipeline|leads|deal)\b",
        flags=re.I,
    )
    sales_context_pattern = re.compile(
        r"\b(store|client|customer|account|territory|region|deal|retail|b2b|b2c)\b",
        flags=re.I,
    )

    data_anchor_pattern = re.compile(
        r"\b(data|dataset|analysis|sql|report|dashboard|insights|etl|pipeline|bi)\b",
        flags=re.I,
    )
    data_context_pattern = re.compile(
        r"\b(company|client|business|dataset|tableau|power bi|snowflake|warehouse)\b",
        flags=re.I,
    )

    pm_anchor_pattern = re.compile(
        r"\b(product|feature|launch|roadmap|mvp|user research|stakeholder|requirements)\b",
        flags=re.I,
    )
    pm_context_pattern = re.compile(
        r"\b(company|team|users|customer|market|launch|release|experiment|sprint)\b",
        flags=re.I,
    )

    # Ask for more concrete context if the candidate is using buzzwords without grounding

    # Software Engineer context anchor
    if role_lower == "software engineer":
        if se_anchor_pattern.search(text_all) and not se_context_pattern.search(text_all):
            key = "anchor_se"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Before going deeper, could you describe a system you worked on that needed "
                    "scale or concurrency—what was the scale and your role?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

    # Sales context anchor
    if role_lower == "sales associate":
        if sales_anchor_pattern.search(text_all) and not sales_context_pattern.search(text_all):
            key = "anchor_sales"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Before we continue, could you describe a sales environment you worked in—"
                    "customer type, product, and your sales targets?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

    # Data Analyst context anchor
    if role_lower == "data analyst":
        if data_anchor_pattern.search(text_all) and not data_context_pattern.search(text_all):
            key = "anchor_data"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Before diving deeper, could you describe a dataset you worked on—its size, "
                    "source, and the goal of the analysis?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

    # Product Manager context anchor
    if role_lower == "product manager":
        if pm_anchor_pattern.search(text_all) and not pm_context_pattern.search(text_all):
            key = "anchor_pm"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Before going further, could you describe a product or feature you worked on—"
                    "target users and business goal?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

    # --------------------------------------------------------
    # Metric-oriented follow-ups (role-specific)
    # --------------------------------------------------------
    metric_terms = re.compile(
        r"\b(%|kpi|quota|target|conversion|retention|coverage|latency|rps|accuracy|precision|revenue|close rate)\b",
        flags=re.I,
    )
    metric_match = metric_terms.search(last_a)
    if metric_match:
        key = f"metric_{metric_match.group(0).lower()}"
        if key not in st.session_state.followups_asked:
            st.session_state.followups_asked.add(key)

            if role_lower == "sales associate":
                q = "How did that affect your close rate or revenue contribution?"
            elif role_lower == "data analyst":
                q = "How did that insight impact business decisions or performance metrics?"
            elif role_lower == "product manager":
                q = "How did that affect user adoption, retention, or revenue impact?"
            else:
                q = "How did it affect reliability, latency, or user experience?"

            st.session_state.last_question = q
            st.session_state.repeat_count = 0
            return q

    # --------------------------------------------------------
    # Role-specific depth checks
    # --------------------------------------------------------

    # Software Engineer depth
    se_primitive_search = re.search(
        r"\b(cas|compare[- ]and[- ]swap|optimistic locking|mutex|spinlock|atomic|lock[- ]free)\b",
        last_a,
        flags=re.I,
    )
    if role_lower == "software engineer" and se_primitive_search:
        prim = se_primitive_search.group(0).lower()
        key = f"se_prim_{prim}"
        if key not in st.session_state.followups_asked:
            st.session_state.followups_asked.add(key)
            q = (
                f"You mentioned {prim}. Why was that preferable over alternatives like a mutex, "
                "and what drawbacks did it introduce?"
            )
            st.session_state.last_question = q
            st.session_state.repeat_count = 0
            return q

    # Sales depth
    sales_depth_search = re.search(
        r"\b(negotiation|objection|closing|pipeline|crm|prospecting)\b",
        last_a,
        flags=re.I,
    )
    if role_lower == "sales associate" and sales_depth_search:
        term = sales_depth_search.group(0).lower()
        key = f"sales_depth_{term}"
        if key not in st.session_state.followups_asked:
            st.session_state.followups_asked.add(key)
            q = (
                f"You mentioned {term}. What specific tactic or approach did you use there, "
                "and how did it influence the outcome?"
            )
            st.session_state.last_question = q
            st.session_state.repeat_count = 0
            return q

    # Data Analyst depth
    data_depth_search = re.search(
        r"\b(model|regression|classification|sql query|join|dashboard|etl)\b",
        last_a,
        flags=re.I,
    )
    if role_lower == "data analyst" and data_depth_search:
        term = data_depth_search.group(0).lower()
        key = f"data_depth_{term}"
        if key not in st.session_state.followups_asked:
            st.session_state.followups_asked.add(key)
            q = (
                f"You mentioned {term}. What trade-offs did you consider regarding accuracy, "
                "performance, or interpretability?"
            )
            st.session_state.last_question = q
            st.session_state.repeat_count = 0
            return q

    # Product Manager depth
    pm_depth_search = re.search(
        r"\b(prioritization|roadmap|launch|mvp|user research|stakeholder)\b",
        last_a,
        flags=re.I,
    )
    if role_lower == "product manager" and pm_depth_search:
        term = pm_depth_search.group(0).lower()
        key = f"pm_depth_{term}"
        if key not in st.session_state.followups_asked:
            st.session_state.followups_asked.add(key)
            q = (
                f"You mentioned {term}. What decision framework did you use there, "
                "and what trade-offs were involved?"
            )
            st.session_state.last_question = q
            st.session_state.repeat_count = 0
            return q

    # --------------------------------------------------------
    # Role-specific scenario / design questions in deep-dive stage
    # --------------------------------------------------------
    if stage == "deep_dive":
        if role_lower == "software engineer":
            key = "se_design"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Design a high-throughput order processing service: outline components, data flow, "
                    "and how you'd ensure throughput and reliability."
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

        if role_lower == "sales associate":
            key = "sales_scenario"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Imagine a key client is hesitating due to pricing—walk me through how you would "
                    "handle that objection and close the deal."
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

        if role_lower == "data analyst":
            key = "data_scenario"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Suppose leadership asks for a dashboard tomorrow using incomplete data—how would you "
                    "balance speed with accuracy and reliability?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

        if role_lower == "product manager":
            key = "pm_scenario"
            if key not in st.session_state.followups_asked:
                st.session_state.followups_asked.add(key)
                q = (
                    "Your team wants to build Feature A, leadership wants Feature B—how would you decide "
                    "and communicate that decision?"
                )
                st.session_state.last_question = q
                st.session_state.repeat_count = 0
                return q

    # --------------------------------------------------------
    # Default: fall back to LLM to generate a follow-up question
    # --------------------------------------------------------
    prompt_lines = [
        "You are a professional interviewer. Produce ONE natural follow-up question (single sentence).",
        f"Role: {role}. Stage: {stage}. Tone: neutral and conversational.",
    ]

    if live:
        parts = []
        if live.get("experience_level"):
            parts.append(f"level={live['experience_level']}")
        if live.get("skills"):
            parts.append("skills=" + ", ".join(live["skills"][:3]))
        if live.get("confidence"):
            parts.append(f"confidence={live['confidence']}")
        if parts:
            prompt_lines.append(
                "Hidden summary (do NOT show to candidate): " + "; ".join(parts) + "."
            )

    last_snippet = short(last.get("a", ""))
    if last_snippet:
        prompt_lines.append(f'Last answer (short): "{last_snippet}".')

    prompt_lines.append("Goal: Ask a meaningful follow-up appropriate to the role.")
    prompt_lines.append(
        "CONSTRAINTS: Output ONLY one question, <= 25 words, no 'prompt', avoid 'Regarding' or 'About'."
    )

    followup_prompt = "\n".join(prompt_lines)

    try:
        resp = model.generate_content(followup_prompt)
        text = (resp.text or "").strip()

        q_line = ""
        for ln in text.splitlines():
            ln = ln.strip()
            if ln:
                q_line = ln
                break

        q_line = q_line.strip().strip('"').strip("'")
        if not q_line.endswith("?"):
            q_line = q_line.rstrip(".") + "?"

        st.session_state.last_question = q_line
        st.session_state.repeat_count = 0
        return q_line
    except Exception:
        fallback = f"What experience best demonstrates your fit for the {role} role and why?"
        st.session_state.last_question = fallback
        st.session_state.repeat_count = 0
        return fallback


# ============================================================
# AI: Final feedback at the end of the interview
# ============================================================

def generate_feedback(qa_list, role, mode, persona):
    """
    Ask Gemini to summarize the interview into a short, structured feedback block.
    """
    prompt = f"""
You are a senior hiring manager and interview coach.

Evaluate the following mock interview.

Role: {role}
Overall inferred candidate persona/style: {persona}

Interview transcript (list of questions, answers, and any detected issues):
{json.dumps(qa_list, indent=2)}

Your response MUST follow EXACTLY this structure (headings and order):

Score: X/10

Strengths:
- ...

Weaknesses:
- ...

Communication:
- ...

Technical Evaluation:
- ...

Behavior & Soft Skills:
- ...


Suggestions (very concrete, short bullet points):
1. ...
2. ...
3. ...

Do not add any other sections. Keep it concise but specific.
"""

    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Could not generate feedback due to an error: {e}"


# ============================================================
# UI: SETUP STAGE
# ============================================================

if st.session_state.stage == "setup":
    st.subheader("Step 1: Interview Settings")

    role = st.selectbox(
        "Select the job role:",
        ["Software Engineer", "Sales Associate", "Data Analyst", "Product Manager"],
    )

    num_questions = st.slider(
        "Number of main questions:",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button("Start Interview"):
        st.session_state.role = role
        # Internally we keep a single mode "Mixed" for now, but keep the field for extension
        st.session_state.mode = "Mixed"
        st.session_state.num_questions = num_questions
        st.session_state.stage = "interview"
        st.session_state.current_question_index = 0
        st.session_state.qa_list = []
        st.session_state.persona = "Balanced"
        st.rerun()

# ============================================================
# UI: INTERVIEW STAGE
# ============================================================

elif st.session_state.stage == "interview":
    st.subheader("Step 2: Mock Interview")

    # Keep persona updated as user answers more questions
    inferred_persona = infer_persona(st.session_state.qa_list)
    st.session_state.persona = inferred_persona

    st.write(f"**Role:** {st.session_state.role}")
    st.write(f"**Detected candidate style (so far):** {inferred_persona}")
    st.write(f"**Total Questions:** {st.session_state.num_questions}")

    idx = st.session_state.current_question_index

    # If we reached the target number of questions, move on to feedback
    if idx >= st.session_state.num_questions:
        st.session_state.stage = "feedback"
        st.rerun()

    # Cache each generated question so that reruns don't regenerate it
    q_key = f"question_{idx}"
    if q_key not in st.session_state:
        q_text = generate_ai_question(
            st.session_state.role,
            st.session_state.mode,
            inferred_persona,
            st.session_state.qa_list,
        )
        st.session_state[q_key] = q_text
    else:
        q_text = st.session_state[q_key]

    st.markdown(f"### Question {idx + 1}")
    st.write(q_text)

    answer = st.text_area(
        "Your answer:",
        key=f"answer_{idx}",
        height=180,
        placeholder="Type your answer here. You can be detailed, and try to use examples.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        submit_btn = st.button("Submit & Next")
    with col2:
        end_btn = st.button("End Interview Now")
    with col3:
        skip_btn = st.button("Skip Question")

    # When user submits an answer, analyze and store it, but don't show per-question feedback
    if submit_btn:
        label, coach = analyze_answer(
            q_text, answer, st.session_state.role, st.session_state.mode
        )

        st.session_state.qa_list.append(
            {
                "q": q_text,
                "a": answer or "[No answer]",
                "issue": label,
                "coach": coach,
            }
        )
        st.session_state.current_question_index += 1
        st.rerun()

    if skip_btn:
        st.session_state.qa_list.append(
            {
                "q": q_text,
                "a": "[Skipped]",
                "issue": "too_short",
                "coach": "You skipped this question. In a real interview, try to always attempt an answer, even if partial.",
            }
        )
        st.session_state.current_question_index += 1
        st.rerun()

    if end_btn:
        # Mark the last answer as incomplete because the interview ended mid-way
        label, coach = analyze_answer(
            q_text, answer, st.session_state.role, st.session_state.mode
        )
        st.session_state.qa_list.append(
            {
                "q": q_text,
                "a": answer or "[No answer]",
                "issue": "incomplete",
                "coach": "Interview ended early. In practice, try to finish the full interview to get a complete evaluation.",
            }
        )
        st.session_state.stage = "feedback"
        st.rerun()

# ============================================================
# UI: FEEDBACK STAGE
# ============================================================

elif st.session_state.stage == "feedback":
    st.subheader("Step 3: AI Feedback Summary")

    overall_persona = infer_persona(st.session_state.qa_list)
    st.markdown(f"**Overall inferred candidate style:** {overall_persona}")

    st.markdown("### 📝 Interview Transcript")
    for i, item in enumerate(st.session_state.qa_list, start=1):
        st.markdown(f"**Q{i}:** {item['q']}")
        st.markdown(f"**A{i}:** {item['a']}")
        if item.get("issue") and item["issue"] != "ok":
            st.markdown(f"_Coaching note_: {item.get('coach', '')}")
        st.markdown("---")

    st.markdown("### 🔍 Overall Evaluation")

    with st.spinner("Gemini is analyzing your performance..."):
        feedback = generate_feedback(
            st.session_state.qa_list,
            st.session_state.role,
            st.session_state.mode,
            overall_persona,
        )

    st.write(feedback)

    if st.button("Start a New Interview"):
        reset_interview()
        st.rerun()
