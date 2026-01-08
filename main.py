import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client
from pypdf import PdfReader
import gspread
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
SERVICE_ACCOUNT_PATH = os.environ.get("GOOGLE_SERVICE_ACCOUNT_PATH")
SHEET_ID_FEEDBACK = os.environ.get("GOOGLE_SHEET_ID_FEEDBACK")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_gc = None
_sheet = None
_feedback_sheet = None

def get_sheets():
    global _gc, _sheet, _feedback_sheet
    if _gc is None:
        try:
            _gc = gspread.service_account(filename=SERVICE_ACCOUNT_PATH)
            _sheet = _gc.open_by_key(SHEET_ID).sheet1
            _feedback_sheet = _gc.open_by_key(SHEET_ID_FEEDBACK).sheet1
        except Exception as e:
            print(f"Error connecting to Google Sheets: {e}")
            raise
    return _sheet, _feedback_sheet

leave_state = {
    "employee_id": None,
    "employee_name": None,
    "leave_type": None,
    "start_date": None,
    "end_date": None,
    "number_of_days": None,
    "comments": None
}

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk(text, size=200, overlap=100):
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def embed(texts):
    if not texts:
        return []
    try:
        res = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [e.embedding for e in res.data]
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []

def store(chunks, embeddings):
    if not chunks or not embeddings:
        return
    try:
        supabase.table("pydoc").delete().neq("id", 0).execute()
        for c, e in zip(chunks, embeddings):
            supabase.table("pydoc").insert({
                "content": c,
                "metadata": {},
                "embedding": e
            }).execute()
    except Exception as e:
        print(f"Error storing embeddings: {e}")

def retrieve(query):
    if not query:
        return ""
    try:
        q_emb = embed([query])[0]
        res = supabase.rpc("match_pydoc", {
            "query_embedding": q_emb,
            "match_count": 4,
            "filter": {}
        }).execute()
        return " ".join([r["content"] for r in res.data]) if res.data else ""
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def policy_chat(question):
    try:
        context = retrieve(question)
        if not context:
            return "I don't have information about that policy. Please contact HR directly."
        
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question directly and concisely using only the provided context. If the information is not in the context, say you do not have that information."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Error in policy chat: {e}")
        return "I encountered an error processing your policy question. Please try again."

def leave_agent(message, current_state):
    system_prompt = f"""
HR leave assistant. Today: {datetime.now().strftime('%Y-%m-%d')}

Extract info from message and current_state. Preserve existing values.

Return JSON:
{{
    "status": "missing|complete",
    "updated_state": {{"employee_id": null, "employee_name": null, "leave_type": null, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "number_of_days": int, "comments": ""}},
    "message": "conversational response",
    "leave_data": {{all fields or null}}
}}

EXTRACTION:
- employee_id: any 3+ digits → extract as string
- employee_name: "I am/my name is X" or just a name
- leave_type: INFER from comments/message
  * sick/ill/doctor/medical → "sick"
  * family/vacation/trip/holiday/eid → "vacation"
  * personal/urgent/emergency/wedding → "casual"
  * default → "vacation"
- dates: Parse flexibly:
  * "tomorrow" → calculate from today
  * "8 Jan 2026", "Jan 8", "2026-01-08" → parse to YYYY-MM-DD
  * "from X to Y" → extract both dates
  * Relative: "next Monday", "in 3 days"
- duration: "X days" (calculate end_date if start_date exists)
- comments: reason/explanation

AUTO-CALCULATE:
- start_date + number_of_days → end_date
- start_date + end_date → number_of_days
- If only start_date provided, default to 1 day

PRESERVE EXISTING:
- Keep all non-null values from current_state
- Only update with new information from message
- NEVER reset a field that was already set

DATE PARSING EXAMPLES:
- "tomorrow" → {datetime.now() + timedelta(days=1):%Y-%m-%d}
- "8 Jan 2026" → "2026-01-08"
- "Jan 8" → assume current year
- "in 2 days" → calculate from today

MESSAGES (natural, friendly):
- Missing Everything: "I can help you apply for leave! Could you tell me your employee ID and name?"
- Missing id+name: "What's your employee ID and name?"
- Missing id: "Thanks! What's your employee ID?"
- Missing name: "And what's your name?"
- Missing dates: "When would you like to take leave?"
- Missing start: "What's the start date?"
- Missing end: "And the end date (or how many days)?"
- Missing leave_type: "What's the reason for your leave?"

REQUIRED: employee_id, employee_name, leave_type, start_date, end_date, number_of_days
All present → "complete", else → "missing"

IMPORTANT: If you receive a date in ANY format, convert it to YYYY-MM-DD and update start_date immediately.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current State: {json.dumps(current_state)}\n\nNew Message: {message}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        data = json.loads(response.choices[0].message.content)
        
        
        if "updated_state" not in data:
            data["updated_state"] = current_state.copy()
        else:
            merged = current_state.copy()
            for key, value in data["updated_state"].items():
                if value is not None:
                    merged[key] = value
            data["updated_state"] = merged
        
        required = ["employee_id", "employee_name", "leave_type", "start_date", "end_date", "number_of_days"]
        updated = data["updated_state"]
        
        is_complete = all(updated.get(f) for f in required)
        
        if is_complete:
            data["status"] = "complete"
            data["leave_data"] = {
                "employee_id": str(updated["employee_id"]),
                "employee_name": str(updated["employee_name"]),
                "leave_type": str(updated["leave_type"]),
                "start_date": str(updated["start_date"]),
                "end_date": str(updated["end_date"]),
                "number_of_days": int(updated["number_of_days"]),
                "comments": str(updated.get("comments", ""))
            }
        else:
            data["status"] = "missing"
            data["leave_data"] = None
        
        return data
        
    except Exception as e:
        print(f"[ERROR] leave_agent exception: {e}")
        return {
            "status": "missing",
            "updated_state": current_state,
            "message": "Sorry, I didn't catch that. Could you repeat?",
            "leave_data": None
        }

def submit_leave(data):
    try:
        sheet, _ = get_sheets()
        sheet.append_row([
            str(data.get("employee_id", "")),
            str(data.get("employee_name", "")),
            str(data.get("leave_type", "")),
            str(data.get("start_date", "")),
            str(data.get("end_date", "")),
            str(data.get("number_of_days", "")),
            "Pending",
            datetime.now().strftime("%Y-%m-%d"),
            "",
            str(data.get("comments", ""))
        ])
    except Exception as e:
        print(f"Error submitting leave: {e}")
        raise

def feedback_agent(message):
    system_prompt = """
You are a feedback processing assistant. Extract and structure user feedback into actionable insights.

Return ONLY valid JSON with these exact fields:
{
    "feedback": "the user's complete feedback as a single string",
    "sentiment": "Positive, Neutral, or Negative",
    "action_items": "brief summary of actions needed as a single string"
}

SENTIMENT CLASSIFICATION:
- Positive: praise, appreciation, satisfaction (e.g., "good", "love", "great", "excellent")
- Negative: complaints, dissatisfaction, problems (e.g., "don't like", "bad", "poor", "issues")
- Neutral: suggestions, observations, questions without strong emotion

ACTION ITEM INFERENCE GUIDE:

For POSITIVE feedback:
- Pattern: User expresses satisfaction with something
- Action: "Continue [current practice]" or "Maintain [what's working]"
- Example: "breakfast is early" → "Continue scheduling breakfast at the current early time"

For NEGATIVE feedback (complaints/problems):
- Pattern: User dislikes something specific
- Action: "Investigate and address [specific issue]" or "Conduct survey on [topic]"
- Example: "don't like the food" → "Conduct a survey to gather specific feedback on food preferences"

- Pattern: User mentions lack/insufficiency
- Action: "Increase [what's lacking]" or "Expand [limited resource]"
- Example: "no multiple options" → "Increase the variety of breakfast options available"

- Pattern: User reports quality issues
- Action: "Improve quality of [item]" or "Review and enhance [service]"
- Example: "food is cold" → "Review food temperature management and serving procedures"

For NEUTRAL feedback (suggestions):
- Pattern: User proposes ideas without complaint
- Action: "Consider [suggestion]" or "Evaluate feasibility of [idea]"
- Example: "add tea option" → "Consider adding tea to the beverage menu"

GENERAL RULES:
- Be specific and actionable (avoid vague responses like "address feedback")
- Match action urgency to sentiment intensity
- If feedback is too vague: "Seek clarification on specific concerns"
- If no clear action needed: "Acknowledge and monitor for patterns"
- Keep action_items concise (1-2 sentences max)

FIELD REQUIREMENTS:
- feedback: single string capturing complete user input
- sentiment: exactly one word from [Positive, Neutral, Negative]
- action_items: single string (not a list), specific and actionable
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(response.choices[0].message.content)
        
        if isinstance(data.get("action_items"), list):
            data["action_items"] = "; ".join(str(item) for item in data["action_items"])
        
        if isinstance(data.get("feedback"), list):
            data["feedback"] = " ".join(str(item) for item in data["feedback"])
        
        data.setdefault("feedback", message)
        data.setdefault("sentiment", "neutral")
        data.setdefault("action_items", "No specific action required")
        
        return data
    except Exception as e:
        print(f"Error in feedback agent: {e}")
        return {
            "feedback": message,
            "sentiment": "neutral",
            "action_items": "No specific action required"
        }

def submit_feedback(data):
    try:
        _, feedback_sheet = get_sheets()
        feedback_sheet.append_row([
            str(data.get("feedback", ""))[:1000],
            str(data.get("sentiment", "neutral")),
            str(data.get("action_items", "No action required"))[:500]
        ])
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        raise

def universal_agent(message, current_context=None):
    if not message or not message.strip():
        return ""
    
    context_hint = ""
    if current_context and current_context.get("active_flow"):
        context_hint = f"\n\nCONTEXT: User is currently in the middle of a {current_context['active_flow']} flow."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   
            messages=[ 
                { 
                    "role": "system", 
                    "content": (
                        "You are a STRICT intent classification engine.\n"
                        "Classify user intent into EXACTLY ONE label.\n\n"
                        "LABELS:\n\n"
                        "POLICY - Questions about company rules, policies, leave balance, eligibility, types of leave\n\n"
                        "LEAVE - User wants to apply for or take leave (direct or indirect requests) OR providing information during leave application\n\n"
                        "FEEDBACK - User giving feedback, complaints, suggestions, or sharing experiences\n\n"
                        "RULES:\n"
                        "1. If currently in LEAVE flow and user provides employee details, dates, or leave info → LEAVE\n"
                        "2. If user mentions 'feedback', 'complaint', 'issue', 'suggestion' explicitly → FEEDBACK\n"
                        "3. If user asks HOW or WHAT about policies → POLICY\n"
                        "4. If user requests/plans leave OR provides leave-related info → LEAVE\n"
                        "5. Employee IDs, names, dates in response to leave questions → LEAVE\n"
                        "6. Return ONLY one word: POLICY, LEAVE, or FEEDBACK\n"
                        "7. No explanations, no punctuation, no extra text\n"
                        + context_hint
                    )
                }, 
                { 
                    "role": "user", 
                    "content": message 
                }
            ],
            max_tokens=10,
            temperature=0
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        valid_intents = ["POLICY", "LEAVE", "FEEDBACK"]
        if intent not in valid_intents:
            for valid in valid_intents:
                if valid in intent:
                    return valid
            return ""
        
        return intent
    except Exception as e:
        print(f"Error in intent classification: {e}")
        return ""

if __name__ == "__main__":
    print("HR Assistant ready.")
    print("Type 'exit' to quit.\n")
    
    conversation_context = {"active_flow": None}
    
    while True:
        try:
            user_input = input(">> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            intent = universal_agent(user_input, conversation_context)
            
            if not intent:
                print("I couldn't understand your request. I can help with leave applications, policy questions, or feedback.")
                conversation_context["active_flow"] = None
                continue
            
            print(f"[DEBUG] Classified intent: {intent}")

            if intent == "POLICY":
                conversation_context["active_flow"] = None
                response = policy_chat(user_input)
                print(response)
                continue

            if intent == "FEEDBACK":
                conversation_context["active_flow"] = None
                try:
                    feedback = feedback_agent(user_input)
                    submit_feedback(feedback)
                    print("Thank you for your feedback! It has been recorded.")
                except Exception as e:
                    print(f"Sorry, I couldn't record your feedback. Please try again later. Error: {e}")
                continue

            if intent == "LEAVE":
                conversation_context["active_flow"] = "LEAVE"
                result = leave_agent(user_input, leave_state)

                if result.get("status") == "missing":
                    leave_state.update(result.get("updated_state", {}))
                    print(result.get("message", "Please provide more information."))
                    continue

                if result.get("status") == "complete":
                    try:
                        submit_leave(result["leave_data"])
                        print(
                            f"✓ Leave submitted successfully!\n"
                            f"  Days: {result['leave_data']['number_of_days']}\n"
                            f"  Period: {result['leave_data']['start_date']} → {result['leave_data']['end_date']}\n"
                            f"  Status: Pending approval"
                        )
                        leave_state = {k: None for k in leave_state}
                        conversation_context["active_flow"] = None
                    except Exception as e:
                        print(f"Failed to submit leave. Please try again. Error: {e}")
                    continue

            print("Sorry, I'm not capable of handling this request. I can help with leave applications, policy questions, or feedback.")
            conversation_context["active_flow"] = None
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please try again.")