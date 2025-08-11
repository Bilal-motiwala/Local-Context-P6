import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
    trace,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env file.")

# ---------------------------
# OpenAI Client Setup
# ---------------------------
external_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",  # Fast & cheap model
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client
)

# ---------------------------
# Local Context Classes
# ---------------------------
class BankAccount:
    def __init__(self, account_number, customer_name, account_balance, account_type):
        self.account_number = account_number
        self.customer_name = customer_name
        self.account_balance = account_balance
        self.account_type = account_type

class StudentProfile:
    def __init__(self, student_id, student_name, current_semester, total_courses):
        self.student_id = student_id
        self.student_name = student_name
        self.current_semester = current_semester
        self.total_courses = total_courses

class LibraryBook:
    def __init__(self, book_id, book_title, author_name, is_available):
        self.book_id = book_id
        self.book_title = book_title
        self.author_name = author_name
        self.is_available = is_available

# ---------------------------
# Create Local Context Objects
# ---------------------------
bank_account = BankAccount(
    account_number="ACC-789456",
    customer_name="Fatima Khan",
    account_balance=75500.50,
    account_type="savings"
)

student = StudentProfile(
    student_id="STU-456",
    student_name="Hassan Ahmed",
    current_semester=4,
    total_courses=5
)

library_book = LibraryBook(
    book_id="BOOK-123",
    book_title="Python Programming",
    author_name="John Smith",
    is_available=True
)

# ---------------------------
# Guardrail Model
# ---------------------------
class MedicineOutput(BaseModel):
    response: str
    isMedicineQuery: bool

# Guardrail Agent
guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="""
    You are a guardrail agent. Your job is to check if the user question is about medicine.
    Return:
    {
        "response": "<short message>",
        "isMedicineQuery": true/false
    }
    """,
    output_type=MedicineOutput
)

# Guardrail Function
@input_guardrail
async def medicine_guardrail(ctx, agent, input_text) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input_text, run_config=config)
    output = result.final_output
    is_medicine = False

    if isinstance(output, MedicineOutput):
        is_medicine = output.isMedicineQuery
    elif isinstance(output, dict):
        is_medicine = bool(output.get("isMedicineQuery", False))

    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=not is_medicine
    )

# ---------------------------
# Agents (with Local Context)
# ---------------------------
local_context_text = f"""
Bank Account:
  Account Number: {bank_account.account_number}
  Customer Name: {bank_account.customer_name}
  Account Balance: {bank_account.account_balance}
  Account Type: {bank_account.account_type}

Student Profile:
  ID: {student.student_id}
  Name: {student.student_name}
  Semester: {student.current_semester}
  Total Courses: {student.total_courses}

Library Book:
  ID: {library_book.book_id}
  Title: {library_book.book_title}
  Author: {library_book.author_name}
  Available: {library_book.is_available}
"""

medicine_agent = Agent(
    name="Medicine Agent",
    instructions=f"""
    You are a medicine expert. Answer user questions about medicine clearly and concisely.
    Also, you have access to the following local context data:
    {local_context_text}
    If the question is not about medicine, do not answer.
    """
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=f"""
    You are a triage agent. Use the given local context:
    {local_context_text}
    Delegate the user's query to the correct agent. Use medicine_agent for medicine-related queries.
    """,
    handoffs=[medicine_agent],
    input_guardrails=[medicine_guardrail]
)

# ---------------------------
# Runner
# ---------------------------
async def run_query(user_query: str):
    with trace("Guardrail Execution"):
        try:
            result = await Runner.run(triage_agent, user_query, run_config=config)
            print("\n--- Agent Answer ---")
            print(result.final_output)
            print("--------------------\n")
        except InputGuardrailTripwireTriggered:
            print("\n[Blocked] ❌ This question is not about medicine.\n")

def main():
    print("Medicine Query Bot (Guardrails + Local Context Enabled)")
    print("Type 'exit' to quit.\n")
    loop = asyncio.get_event_loop()

    while True:
        question = input("Your Question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if question:
            loop.run_until_complete(run_query(question))

if __name__ == "__main__":
    main()
