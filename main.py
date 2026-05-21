# ============================================================
# Slumdog Pythonair Agent — SOLUTION
# ============================================================

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
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
llm = ChatOllama(model="llama3.2:1b", temperature=0.4)


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
        "game_over": False
    }


def fetch_question_node(state: TriviaState) -> dict:
    # TODO #1 — SOLUTION
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
    return {
        "question": question,
        "correct_answer": correct,
        "choices": choices
    }


def present_question_node(state: TriviaState) -> dict:
    # present the question to the user
    print(f"\nQuestion {state['total_questions'] + 1}:")
    print(f"  {state['question']}\n")
    for i, choice in enumerate(state['choices'], 1):
        print(f"  {i}. {choice}")
    # Request user input
    answer = input("\nYour answer (1–4, or 'q' to quit): ").strip()

    # In case the user want to exit
    if answer.lower() == 'q':
        return {"game_over": True, "user_answer": ""}

    try:
        selected = state['choices'][int(answer) - 1]
        print(f"You selected: {selected}")
    except (ValueError, IndexError):
        print("Invalid input, marking as wrong.")
        selected = ""
    # Return the state for the next node
    return {
        "user_answer": selected,
        "total_questions": state['total_questions'] + 1
    }


def judge_node(state: TriviaState) -> dict:
    # TODO #2 — SOLUTION
    prompt = f"""
    Using the following answers:
    The trivia question was: {state['question']}
    The correct answer is: {state['correct_answer']}
    The player answered: {state['user_answer']}

    tell the player in 1-2 sentences, if they got it right or wrong. Have the correct answer provided 
    above as ground truth.

    Be friendly and cheerful, replying directly to the player!

    In case you have any additional information rearding the question,
    share some details, after telling the user the result.
    """
    # Invoke the LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n {response.content}")

    # Update score
    if state['user_answer'].strip().lower() == state['correct_answer'].strip().lower():
        is_correct = True
    new_score = state['score'] + (1 if is_correct else 0)

    print(f"\n Score: {new_score} / {state['total_questions']}")

    # Prompt the user to continue
    again = input("\nNext question? (y/n): ").strip().lower()
    if again != "y":
        game_over = True
    return {"score": new_score, "game_over": game_over}

# Routing functions
def route_after_judge(state: TriviaState) -> str:
    # TODO #3 — SOLUTION
    if state["game_over"]:
        return "end"
    return "fetch_question"


def route_after_present_question(state: TriviaState) -> str:
    # THIS IS TO BE PASSED AS A BUG the user should solve
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
    graph.add_conditional_edges("present_question",
                               route_after_present_question,
                               {
                                   "judge":"judge",
                                   "end": END
                               })

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "fetch_question": "fetch_question",
            "end": END
        }
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
    print(f"\n🏁 Game over! Final score: {final_state['score']}/{final_state['total_questions']}")