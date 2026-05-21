# Slumdog Pythonair — `main.py` (Reference Solution)

Slumdog Pythonaire: Building a trivia game using Python and LangGraph
Today you will build your own app that can quiz you, judge your answers, and roast you when you get them wrong (how fun 😒)! 
In this workshop, we'll build together a fully working trivia game powered by an LLM model, from scratch (well, almost!). 

You'll learn how to fetch trivia questions from the web, feed them into an AI agent that thinks and responds, as well as keep track of your score as you go!
We'll be using LangGraph, a tool that lets you design your AI's "brain" as a flow, as well as Ollama, a free tool that lets you run LLMs on your machine.

By the end of this workshop you will have:
- Made an API call to fetch trivia questions 
- Built an AI agent using LangGraph
- Learned about tools that will help you build your own AI powered projects!

```bash
uv run main.py
```

---

## 1. Prerequisites

You'll need:
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package & env manager
- **[Ollama](https://ollama.com/download)** — to run a small LLM locally
- A working internet connection (for trivia questions and the first model pull)

---

## 2. What `main.py` does

`main.py` builds a small **state machine** (a graph) using
[LangGraph](https://langchain-ai.github.io/langgraph/). Each step of the game is
a *node*, and the game state flows from one node to the next until the game ends.

The game loop is:

1. **Setup** → pick a difficulty.
2. **Fetch a question** → call the Open Trivia DB API.
3. **Present the question** → show choices and read your answer.
4. **Judge** → an LLM tells you if you're right and updates your score.
5. Loop back to step 2, or end.

---

## 3. The game state

Everything the game needs is held in a single `TriviaState` object that gets
passed between nodes. Using a `TypedDict` gives each field a clear type:

```python
class TriviaState(TypedDict):
    score: int             # how many questions you've gotten right
    total_questions: int   # how many you've been asked
    question: str          # the current question text
    choices: list          # the shuffled multiple-choice options
    correct_answer: str    # the right answer (ground truth)
    user_answer: str       # what you selected
    difficulty: str        # easy / medium / hard
    game_over: bool         # flag that ends the loop
```

The LLM itself is created once at module load:

```python
llm = ChatOllama(model="llama3.2:1b", temperature=0.4)
```

This points at Ollama running locally and uses the small, fast `llama3.2:1b`
model. The low `temperature` keeps the judging answers focused.

---

## 4. The nodes

Each node is a plain function that takes the current state and returns a
dictionary of fields to update.

### `setup_node`
Greets the player and asks for a difficulty (`easy` / `medium` / `hard`).
Invalid input defaults to `easy`. Initializes `score`, `total_questions`, and
`game_over`.

### `fetch_question_node`
Builds an Open Trivia DB API URL for the chosen difficulty, requests one
multiple-choice question, and cleans it up:
- `html.unescape(...)` decodes HTML entities (e.g. `&quot;` → `"`).
- The correct and incorrect answers are combined into one `choices` list and
  `random.shuffle`-d so the right answer isn't always in the same spot.

### `present_question_node`
Prints the question and its numbered choices, then reads your input. Typing `q`
quits the game (`game_over = True`). Otherwise it records your selected answer
and increments `total_questions`. Bad input is marked as wrong.

### `judge_node`
Builds a prompt containing the question, the correct answer (as ground truth),
and your answer, then calls the LLM to give a friendly verdict in 1–2 sentences.
It updates the score, prints the running total, and asks whether to continue —
answering anything other than `y` sets `game_over = True`.

---

## 5. The routing functions

After certain nodes, the graph needs to decide where to go next. These small
functions return a label that the graph maps to the next node:

- **`route_after_present_question`** — if you quit, go to the end; otherwise go
  to `judge`.
- **`route_after_judge`** — if the game is over, go to the end; otherwise loop
  back to `fetch_question` for another round.

---

## 6. Building and running the graph

`build_graph()` wires everything together:

```text
setup → fetch_question → present_question ─┬─(quit)──────────────→ END
                                           └─(answer)→ judge ─┬─(continue)→ fetch_question
                                                              └─(stop)────→ END
```

- Nodes are registered with `graph.add_node(...)`.
- Fixed transitions use `graph.add_edge(...)`.
- The two decision points use `graph.add_conditional_edges(...)`, mapping the
  labels returned by the routing functions to actual nodes (or `END`).
- `graph.compile()` produces a runnable app.

The `__main__` block creates an initial empty state, calls `app.invoke(...)` to
run the whole loop, and prints the final score when the game ends.

---

## 7. Run it

```bash
uv run main.py
```

You'll be asked for a difficulty, then served questions one at a time until you
quit or choose not to continue.

---

## Troubleshooting

- **`ConnectionError` to `localhost:11434`**
  Ollama isn't running. Start it with `ollama serve` (or restart the Ollama app).

- **Model not found / `model "llama3.2:1b" not available`**
  Run `ollama pull llama3.2:1b` again.

