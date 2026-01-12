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
            temperature=0,
            max_tokens=2000
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Error in policy chat: {e}")
        return "I encountered an error processing your policy question. Please try again."

def leave_agent(message, current_state):
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after_tomorrow = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

    system_prompt = f"""Extract leave information and merge with existing state. Today: {today}

CURRENT STATE (preserve non-null values):
{json.dumps(current_state)}

EXTRACTION RULES:
1. Extract & merge new info with current state - PRESERVE existing non-null values
2. employee_id: any 3+ digit number
3. employee_name: extract any name mentioned
4. leave_type: "sick"/"casual"/"vacation" - infer if unclear or not mentioned as casual
5. start_date: Parse dates to YYYY-MM-DD format. IMPORTANT date parsing rules:
   - Explicit dates: "Jan 8", "8th Jan", "2026-01-15" → YYYY-MM-DD
   - Standalone "today" or "from today" → {today}
   - Standalone "tomorrow" → {tomorrow}
   - "day after tomorrow" → {day_after_tomorrow}
   - If current state is missing start_date and user just says "today"/"tomorrow", treat it as the start_date
   - Never leave start date as null aslways ask it from the user
   - DO NOT assume or infer start_date from context - it must be explicitly stated by the user 
6. number_of_days: extract from "3 days", "for 5 days", "three", etc.
7. comments: optional additional context

AUTO-CALCULATION (apply after extraction):
- Have start_date + number_of_days? → Calculate end_date = start_date + (number_of_days - 1)
- Have start_date + end_date? → Calculate number_of_days = (end_date - start_date) + 1
- Missing both end_date AND number_of_days? → Ask user for either end_date or number_of_days

IMPORTANT: Return the output as a JSON object with the following format:
{{
  "employee_id": "string or null",
  "employee_name": "string or null",
  "leave_type": "sick|casual|vacation or null",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "number_of_days": number or null,
  "comments": "string or null"
}}

Return COMPLETE merged state (current + new) as JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1000
        )

        details = json.loads(response.choices[0].message.content)

        missing_fields = []
        if not details.get('employee_id'):
            missing_fields.append('employee ID')
        if not details.get('employee_name'):
            missing_fields.append('employee name')
        if not details.get('leave_type'):
            missing_fields.append('leave type (sick/casual/vacation)')
        if not details.get('start_date'):
            missing_fields.append('start date')

        # Check if EITHER end_date OR number_of_days is provided (not both required)
        if not details.get('end_date') and not details.get('number_of_days'):
            missing_fields.append('either end date or number of days')

        if missing_fields:
            return {
                "status": "missing",
                "updated_state": details,
                "message": "I need more information. Please provide:\n- " + "\n- ".join(missing_fields),
                "leave_data": None
            }

        return {
            "status": "complete",
            "updated_state": details,
            "message": "Leave request ready to submit!",
            "leave_data": {
                "employee_id": str(details['employee_id']),
                "employee_name": str(details['employee_name']),
                "leave_type": str(details['leave_type']),
                "start_date": str(details['start_date']),
                "end_date": str(details['end_date']),
                "number_of_days": int(details['number_of_days']),
                "comments": str(details.get('comments', ''))
            }
        }

    except Exception as e:
        print(f"[ERROR] leave_agent exception: {e}")
        return {
            "status": "missing",
            "updated_state": current_state,
            "message": "Sorry, I didn't understand. Could you please repeat?",
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
Extract feedback to JSON: {{"feedback": "string", "sentiment": "Positive|Neutral|Negative", "action_items": "string"}}

Sentiment: Positive (praise/satisfaction), Negative (complaints/problems), Neutral (suggestions/observations)

Actions:
- Positive: "Continue [practice]" or "Maintain [what works]"
- Negative dissatisfaction: "Investigate/address [issue]" or "Survey [topic]"
- Negative lack: "Increase/expand [resource]"
- Negative quality: "Review/enhance [service]"
- Neutral: "Consider/evaluate [suggestion]"
- Vague: "Seek clarification"
- None: "Monitor patterns"

Keep actions specific, concise (1-2 sentences).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1500
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
    print("Welcome to Edra HR Assistant, You can ask me Policy Queries, Leave Applications, or Directly write your Feedback")
    
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