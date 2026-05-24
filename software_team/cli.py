import os
import sys
import argparse
from typing import TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. SHARED MEMORY (STATE)
# ==========================================
class TeamState(TypedDict):
    user_requirements: str
    existing_infrastructure: str
    repo_paths: dict
    business_logic_map: str
    clarification_questions: str
    human_answers: str
    architecture_map: str
    current_code: str
    qa_feedback: str

# ==========================================
# 2. FILE SYSTEM TOOLS (THE AI'S HANDS & EYES)
# ==========================================
@tool
def read_local_file(file_path: str) -> str:
    """Reads the content of a local file. Use this to inspect code before modifying it."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_local_file(file_path: str, content: str) -> str:
    """Writes new code back to a local file. Use this to save refactored code."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def search_directory(directory_path: str, search_term: str) -> str:
    """Searches a directory for a specific text string. Returns a list of file paths containing the text."""
    matches = []
    for root, _, files in os.walk(directory_path):
        if any(skip in root for skip in ["node_modules", ".git", ".nuxt", "vendor"]):
            continue
            
        for file in files:
            if not file.endswith(('.js', '.vue', '.ts', '.php', '.py', '.lua')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if search_term in f.read():
                        matches.append(file_path)
            except Exception:
                pass
                
    if not matches:
        return f"No files found containing '{search_term}'."
    return f"Found '{search_term}' in the following files:\n" + "\n".join(matches[:10])

file_tools = [read_local_file, write_local_file, search_directory]

# ==========================================
# 3. LLM INITIALIZATION
# ==========================================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("\n❌ FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
    print("👉 Fix this by running: export GEMINI_API_KEY='your_api_key_here'")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    api_key=api_key
)
agentic_llm = llm.bind_tools(file_tools)

# ==========================================
# 4. THE AGENTS
# ==========================================
def pm_agent(state: TeamState):
    print("\n🧑‍💼 [PM Agent] is mapping the business logic...")
    
    requirements = state.get("user_requirements", "")
    existing_map = state.get("business_logic_map", "None yet.")
    human_answers = state.get("human_answers", "")

    pm_prompt = f"""You are the lead Product Manager for a software agency.
    Create a 'Business Logic Map' based on user requirements.
    
    PRAGMATISM RULE: Assess the complexity of the task. If it is a simple code cleanup (like removing console.logs), find-and-replace, or trivial script task, DO NOT overcomplicate it. Keep the map very brief and output 'None' for questions. Save your detailed architecture questions for complex feature development.

    If anything is ambiguous on a complex task, write clarification questions. If perfect, write exactly 'None' in the QUESTIONS section.

    Requirements: {requirements}
    User's Previous Answers: {human_answers}
    Current Map: {existing_map}

    Format exactly like this:
    BUSINESS_MAP:
    [Detailed map]
    
    QUESTIONS:
    [Questions or 'None']
    """

    response = llm.invoke([HumanMessage(content=pm_prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    try:
        map_part = content.split("QUESTIONS:")[0].replace("BUSINESS_MAP:", "").strip()
        questions_part = content.split("QUESTIONS:")[1].strip()
    except Exception:
        map_part = content
        questions_part = "None"

    return {"business_logic_map": map_part, "clarification_questions": questions_part}

def human_agent(state: TeamState):
    print("\n" + "="*50)
    print("📝 CURRENT BUSINESS MAP:\n", state["business_logic_map"])
    print("="*50)
    print("\n❓ THE PM HAS QUESTIONS FOR YOU:\n", state["clarification_questions"])
    
    user_input = input("\n👉 Type your answers (or type 'approve' to force the team to start coding): ")
    
    override_words = ["looks good", "approve", "force", "just do it", "all of them", "fuck"]
    if any(word in user_input.lower() for word in override_words):
        return {"clarification_questions": "none", "human_answers": "Approved by user."}
    
    return {"human_answers": user_input}

def team_lead_agent(state: TeamState):
    print("\n🏗️ [Team Lead] is verifying architecture constraints...")
    
    business_map = state.get("business_logic_map", "")
    existing_infra = state.get("existing_infrastructure", "") 

    lead_prompt = f"""You are the Principal Engineer. 
    Design a scalable, pragmatic technical architecture constraint map.
    CRITICAL: You are integrating new requirements into existing infrastructure.
    
    Existing Infrastructure: {existing_infra}
    Business Logic Map: {business_map}

    Format exactly like this:
    ARCHITECTURE_MAP:
    [Your constraints and technical decisions]
    """

    response = llm.invoke([HumanMessage(content=lead_prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    arch_map = content.replace("ARCHITECTURE_MAP:", "").strip()
    return {"architecture_map": arch_map}

def dev_1_agent(state: TeamState):
    print("\n💻 [Senior Dev 1] is thinking deeply and planning the surgical approach...")
    
    business_map = state.get("business_logic_map", "")
    arch_map = state.get("architecture_map", "")
    repo_paths = state.get("repo_paths", {})
    qa_feedback = state.get("qa_feedback", "No previous QA feedback.") # <--- NEW: Grab QA notes if rejected!

    dev_1_prompt = f"""You are an Elite Senior Developer following Andrej Karpathy's engineering philosophy.
    
    YOUR CORE SKILLS:
    1. THINK BEFORE CODING: Do not write code immediately. Use `search_directory` and `read_local_file` to map out what needs to change.
    2. SIMPLICITY FIRST: Write the dumbest, clearest code that solves the problem. No over-engineering.
    
    Target Repositories: {repo_paths}
    Business Logic: {business_map}
    Architecture Constraints: {arch_map}
    Previous QA Feedback (Fix these if present!): {qa_feedback} 
    
    EXECUTION:
    1. Search and read the target files.
    2. Output a brief <THINKING> block explaining your plan.
    3. Draft the initial refactored code using markdown blocks. 
    """

    messages = [HumanMessage(content=dev_1_prompt)]
    response = agentic_llm.invoke(messages)
    
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            print(f"   🔍 Dev 1 requested tool: {tool_call['name']}")
            
            if tool_call["name"] == "read_local_file":
                tool_output = read_local_file.invoke(tool_call["args"])
            elif tool_call["name"] == "search_directory":
                tool_output = search_directory.invoke(tool_call["args"])
            else:
                tool_output = "Tool not allowed."
                
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
        response = agentic_llm.invoke(messages)

    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    return {"current_code": content}

def dev_2_agent(state: TeamState):
    print("\n🕵️‍♂️ [Senior Dev 2] is verifying simplicity and making surgical file updates...")
    
    arch_map = state.get("architecture_map", "")
    draft_code = state.get("current_code", "")
    repo_paths = state.get("repo_paths", {})

    dev_2_prompt = f"""You are an Elite Senior Reviewer and Executor. You pair-program with Dev 1.
    
    YOUR CORE SKILLS:
    1. SURGICAL CHANGES: Never rewrite a whole file unnecessarily. Preserve existing logic perfectly.
    2. GOAL-DRIVEN EXECUTION: Ruthlessly evaluate Dev 1's draft. If it meets the goal, use `write_local_file` to save it.
    
    Target Repositories: {repo_paths}
    Architecture Constraints: {arch_map}
    Dev 1's Draft: {draft_code}
    
    EXECUTION:
    1. Review the draft. Ensure it is simple and surgical.
    2. If needed, use `read_local_file` to verify context.
    3. MUST DO: Use `write_local_file` to surgically apply the final code to the local machine.
    4. Output a brief 'REVIEW NOTES:' confirming the exact lines changed.
    """

    messages = [HumanMessage(content=dev_2_prompt)]
    response = agentic_llm.invoke(messages)
    
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            print(f"   💾 Dev 2 requested tool: {tool_call['name']}")
            
            if tool_call["name"] == "write_local_file":
                tool_output = write_local_file.invoke(tool_call["args"])
            elif tool_call["name"] == "read_local_file":
                tool_output = read_local_file.invoke(tool_call["args"])
            elif tool_call["name"] == "search_directory":
                tool_output = search_directory.invoke(tool_call["args"])
            else:
                tool_output = "Tool not allowed."
                
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
        response = agentic_llm.invoke(messages)

    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    return {"current_code": content}

def qa_1_agent(state: TeamState):
    print("\n🕵️‍♀️ [QA 1: The Breaker] is hunting for edge cases and bugs...")
    
    requirements = state.get("user_requirements", "")
    current_code = state.get("current_code", "")

    qa_1_prompt = f"""You are the Lead QA Tester. Your ONLY job is to find flaws in the code.
    Compare the User Requirements to the Draft Code. 
    Look for edge cases, missing error handling, and security flaws.
    
    User Requirements: {requirements}
    Draft Code: {current_code}
    
    Output a brutal, honest list of potential bugs or missed requirements. 
    If it is absolutely perfect, say "No flaws found."
    """

    response = llm.invoke([HumanMessage(content=qa_1_prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    return {"qa_feedback": content}

def qa_2_agent(state: TeamState):
    print("\n⚖️ [QA 2: The Lead] is making the final Pass/Fail decision...")
    
    qa_1_feedback = state.get("qa_feedback", "")
    current_code = state.get("current_code", "")

    qa_2_prompt = f"""You are the QA Manager. You hold the final release authority.
    Read the code and your QA Tester's feedback.
    
    QA Tester Feedback: {qa_1_feedback}
    Draft Code: {current_code}
    
    If the code has flaws that need fixing, output exactly: REJECTED
    Followed by a list of what Dev 1 needs to fix.
    
    If the code is flawless and ready for production, output exactly: APPROVED
    """

    response = llm.invoke([HumanMessage(content=qa_2_prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

    return {"qa_feedback": content}

# ==========================================
# 5. GRAPH ROUTING & WIRING
# ==========================================
def pm_router(state: TeamState):
    questions = state.get("clarification_questions", "")
    if "none" in questions.lower().strip():
        return "Team_Lead"
    return "Human"

def human_router(state: TeamState):
    """If the user forced approval, skip the PM and go straight to the Team Lead."""
    if state.get("human_answers", "") == "Approved by user.":
        return "Team_Lead"
    return "PM"

def qa_router(state: TeamState):
    feedback = state.get("qa_feedback", "")
    if "APPROVED" in feedback.upper():
        print("\n✅ QA APPROVED! The code is going to production.")
        return END
    else:
        print("\n❌ QA REJECTED! Sending back to Dev 1 to fix the bugs...")
        return "Dev_1"

workflow = StateGraph(TeamState)

workflow.add_node("PM", pm_agent)
workflow.add_node("Human", human_agent)
workflow.add_node("Team_Lead", team_lead_agent)
workflow.add_node("Dev_1", dev_1_agent)
workflow.add_node("Dev_2", dev_2_agent)
workflow.add_node("QA_1", qa_1_agent) 
workflow.add_node("QA_2", qa_2_agent) 

workflow.add_edge(START, "PM")
workflow.add_conditional_edges("PM", pm_router)
workflow.add_conditional_edges("Human", human_router)
workflow.add_edge("Team_Lead", "Dev_1")
workflow.add_edge("Dev_1", "Dev_2")
workflow.add_edge("Dev_2", "QA_1")
workflow.add_edge("QA_1", "QA_2")
workflow.add_conditional_edges("QA_2", qa_router)

app = workflow.compile()

# ==========================================
# 6. CLI EXECUTION
# ==========================================
def main():
    print("\n" + "="*50)
    print("🤖 AI SOFTWARE AGENCY INITIALIZED")
    print("="*50)

    parser = argparse.ArgumentParser(description="Run the AI Software Team")
    parser.add_argument("--task", type=str, help="The task you want the team to do", required=True)
    parser.add_argument("--repo", type=str, help="Absolute path to the local repository", required=True)
    
    args = parser.parse_args()

    initial_input = {
        "user_requirements": args.task,
        "existing_infrastructure": f"Working on local repository located at: {args.repo}",
        "repo_paths": {
            "target_repo": args.repo
        }
    }
    
    print(f"\n🚀 Kicking off the workflow for: {args.repo}")
    
    try:
        app.invoke(initial_input)
        print("\n✅ WORKFLOW COMPLETE!")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()