# ============================================================
# Slumdog Pythonair Agent
# ============================================================

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
import requests
import html
import random


# The state of the Graph. This is object that is 'transferred' between the nodes
# TypedDict is a python type, providing type safety
class TriviaState(TypedDict):
    score: int
    total_questions: int
    question: str
    choices: list
    correct_answer: str
    user_answer: str
    difficulty: str
    game_over: bool


# Create the LLM object here. We use Ollama
llm = ChatOllama(model="llama3.2:latest", temperature=0.4)


# Nodes


def setup_node(state: TriviaState) -> dict:
    print("\n Welcome to SlumDog Pythonair!")
    print("─" * 40)
    difficulty = input("Choose difficulty (easy / medium / hard): ").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        print("Invalid input, defaulting to easy.")
        difficulty = "easy"
    print(f"\nStarting {difficulty} trivia...\n")
    # Return the state of the graph
    return {
        "difficulty": difficulty,
        "score": 0,
        "total_questions": 0,
        "game_over": False,
    }


def fetch_question_node(state: TriviaState) -> dict:
    # Construct the endpoint of open trivia db
    url = f"https://opentdb.com/api.php?amount=1&difficulty={state['difficulty']}&type=multiple"
    response = requests.get(url)
    data = response.json()
    result = data["results"][0]

    # Clean the response
    question = html.unescape(result["question"])
    correct = html.unescape(result["correct_answer"])
    incorrect = [html.unescape(a) for a in result["incorrect_answers"]]

    # Gather the available choices and shuffle them
    choices = incorrect + [correct]
    random.shuffle(choices)

    # Return the state, with the correct answer
    return {"question": question, "correct_answer": correct, "choices": choices}


def present_question_node(state: TriviaState) -> dict:
    # present the question to the user
    print(f"\nQuestion {state['total_questions'] + 1}:")
    print(f"  {state['question']}\n")
    for i, choice in enumerate(state["choices"], 1):
        print(f"  {i}. {choice}")
    # Request user input
    answer = input("\nYour answer (1–4, or 'q' to quit): ").strip()

    # In case the user want to exit
    if answer.lower() == "q":
        return {"game_over": True, "user_answer": ""}

    try:
        selected = state["choices"][int(answer) - 1]
        print(f"You selected: {selected}")
    except (ValueError, IndexError):
        print("Invalid input, marking as wrong.")
        selected = ""
    # Return the state for the next node
    return {"user_answer": selected, "total_questions": state["total_questions"] + 1}


# Tools


@tool
def web_search(query: str) -> str:
    """Search the web for extra detsils about a topic.

    Uses DuckDuckGo API (no API key needed) and
    returns a short text, or an empty string if nothing
    useful was found.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
    except (requests.RequestException, ValueError):
        return ""

    # Prefer the abstract
    if data.get("AbstractText"):
        return data["AbstractText"]

    return ""


# Here, we tell the model about the tool. The LLM itself decides whether
# a call to web_search is needed to answer well. This is the actual agency
tools = [web_search]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


def normalize_query(args: dict, fallback: str) -> str:
    """Fetch a clean query string out of a tool call's args."""
    raw = args.get("query", "")
    if isinstance(raw, dict):
        raw = raw.get("value", "")
    raw = str(raw).strip()
    return raw or fallback


def judge_node(state: TriviaState) -> dict:
    prompt = f"""
    You are the cheerful host of a trivia game. Here is this round:

    - Question:        {state['question']}
    - Correct answer:  {state['correct_answer']}
    - Player's answer: {state['user_answer']}

    STEP 1 — Decide if the player is right.
    The player is CORRECT only if their answer refers to the same thing as the
    correct answer (ignore case and small wording differences).

    STEP 2 — Reply directly to the player in 1-2 sentences:
    - If the player is CORRECT: congratulate them warmly. Do NOT call any tool.
    - If the player is WRONG: call the `web_search` tool exactly once, passing
      the trivia question as the search query. Then tell them they were wrong
      (roast them gracefully), reveal the correct answer, and add one
      interesting detail from the search result.

    Only call `web_search` when the player is WRONG.
    """

    # these three lines essentially kill agency. BUT
    # small models have the disadvantage of doing their thing :(
    # So, i am putting a deterministic condition to avoid the call of tools when t
    # there is no need.
    is_correct = (
        state["user_answer"].strip().lower() == state["correct_answer"].strip().lower()
    )

    # Ask the model, it can ask for a tool
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    response = llm_with_tools.invoke(messages)

    if not is_correct:
        # Wrong answer:
        if response.tool_calls:
            messages.append(response)
            for call in response.tool_calls:
                selected_tool = tools_by_name[call["name"]]
                query = normalize_query(
                    call.get("args", {}), fallback=state["question"]
                )
                print(f"\n(searching the web for: {query})")
                result = selected_tool.invoke({"query": query})
                messages.append(
                    ToolMessage(
                        content=result or "No extra information found.",
                        tool_call_id=call["id"],
                    )
                )
            # Feed the tool results back so the model can finish its reply.
            response = llm_with_tools.invoke(messages)
    else:
        # If the model tried to call a tool
        # (and so returned no text), re-ask without tools for a clean message.
        if response.tool_calls or not str(response.content).strip():
            response = llm.invoke(messages)

    print(f"\n {response.content}")

    # Update score using the deterministic check above.
    new_score = state["score"] + (1 if is_correct else 0)

    print(f"\n Score: {new_score} / {state['total_questions']}")

    # Prompt the user to continue
    again = input("\nNext question? (y/n): ").strip().lower()
    if again != "y":
        game_over = True
    else:
        game_over = False
    return {"score": new_score, "game_over": game_over}


# Routing functions
def route_after_judge(state: TriviaState) -> str:
    if state["game_over"]:
        return "end"
    return "fetch_question"


def route_after_present_question(state: TriviaState) -> str:
    if state["game_over"]:
        return "end"
    return "judge"


# Graph
def build_graph():
    # Initiate the graph state
    graph = StateGraph(TriviaState)

    # Create the nodes - using the functions above
    graph.add_node("setup", setup_node)
    graph.add_node("fetch_question", fetch_question_node)
    graph.add_node("present_question", present_question_node)
    graph.add_node("judge", judge_node)

    # Set the edges
    graph.set_entry_point("setup")
    graph.add_edge("setup", "fetch_question")
    graph.add_edge("fetch_question", "present_question")

    # two branches to correctly route the graph
    graph.add_conditional_edges(
        "present_question", route_after_present_question, {"judge": "judge", "end": END}
    )

    graph.add_conditional_edges(
        "judge", route_after_judge, {"fetch_question": "fetch_question", "end": END}
    )
    # Compile the graph
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    # Initial state
    initial_state: TriviaState = {
        "score": 0,
        "total_questions": 0,
        "question": "",
        "choices": [],
        "correct_answer": "",
        "user_answer": "",
        "difficulty": "medium",
        "game_over": False,
    }
    # Invocation of the graph
    final_state = app.invoke(initial_state)
    # Final result
    print(
        f"\nGame over! Final score: {final_state['score']}/{final_state['total_questions']}"
    )
