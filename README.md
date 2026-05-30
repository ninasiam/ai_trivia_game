# Slumdog Pythonair — Οδηγός Workshop 

## Τι θα φτιάξεις

Σήμερα θα δημιουργήσεις τη δική σου εφαρμογή παιχνιδιού trivia που τροφοδοτείται από AI! Η εφαρμογή θα σε ρωτάει ερωτήσεις, θα κρίνει τις απαντήσεις σου και θα κρατάει το σκορ σου.

Μέχρι το τέλος του workshop θα έχεις:
- Κάνει κλήση API για να λαμβάνεις ερωτήσεις trivia από το διαδίκτυο
- Δημιουργήσει έναν AI agent χρησιμοποιώντας LangGraph
- Μάθει για εργαλεία που θα σε βοηθήσουν να φτιάξεις τα δικά σου AI-powered projects!

Θα χρησιμοποιήσουμε **LangGraph** (σχεδιασμός της λογικής του AI ως ροή) και **Ollama** (δωρεάν εκτέλεση LLM τοπικά στον υπολογιστή σου).

## Βήμα 0 — Προαπαιτούμενα

Πριν ξεκινήσεις, πρέπει έχεις εγκατεστημένα:

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — package manager
- **[Ollama](https://ollama.com/download)** — για να τρέχεις LLM τοπικά
- Σύνδεση στο internet (για τις ερωτήσεις trivia και το πρώτο pull μοντέλου)

## Βήμα 1 — Κατανόηση της κατάστασης του παιχνιδιού

Ολόκληρη η κατάσταση του παιχνιδιού αποθηκεύεται σε ένα object `TriviaState` που περνά από κόμβο σε κόμβο:

```python
class TriviaState(TypedDict):
    score: int             # πόσες ερωτήσεις απάντησες σωστά
    total_questions: int   # πόσες ερωτήσεις έχουν γίνει
    question: str          # το κείμενο της ερώτησης
    choices: list          # οι επιλογές 
    correct_answer: str    # η σωστή απάντηση
    user_answer: str       # η επιλογή σου
    difficulty: str        # easy / medium / hard
    game_over: bool        # δείχνει αν τελείωσε το παιχνίδι
```

## Βήμα 2 — Κατανόηση των κόμβων

Κάθε βήμα του παιχνιδιού είναι ένας **κόμβος** — μια συνάρτηση που παίρνει την κατάσταση και επιστρέφει ενημερωμένα τα attributes.

| Κόμβος | Τι κάνει |
|--------|----------|
| `setup_node` | Καλωσορίζει τον παίκτη και ζητά επίπεδο δυσκολίας |
| `fetch_question_node` | Καλεί το Open Trivia DB API και φέρνει μια ερώτηση |
| `present_question_node` | Εμφανίζει την ερώτηση και επιλογές, διαβάζει την απάντησή σου |
| `judge_node` | Το LLM κρίνει την απάντησή σου και ενημερώνει το σκορ |

## Βήμα 3 — Εκτέλεση της εφαρμογής

```bash
uv run main.py
```

## Αντιμετώπιση προβλημάτων

- **`ConnectionError` στο `localhost:11434`** — Το Ollama δεν τρέχει. Ξεκίνησέ το με `ollama serve`.
- **Μοντέλο δεν βρέθηκε** — Τρέξε `ollama pull llama3.2:1b` ξανά.

---
---

# Slumdog Pythonair — Workshop Tutorial

**Slumdog Pythonaire: Building a trivia game using Python and LangGraph**

Welcome! Today you'll build your own AI-powered trivia game that can quiz you, judge your answers, and keep score — all running locally on your machine. By the end of this workshop you'll understand how to wire up an LLM into a real working app.

## What you'll build

A fully working trivia game powered by an LLM, built from scratch (well, almost!). The game fetches live questions from the web, feeds them to an AI agent that thinks and responds, and tracks your score as you go.

**By the end of this workshop you will have:**
- Made an API call to fetch trivia questions
- Built an AI agent using LangGraph
- Learned about tools that will help you build your own AI-powered projects!

```bash
uv run main.py
```

---

## Step 0 — Set up your environment

Before writing any code, make sure you have the following installed:

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — a fast Python package & environment manager
- **[Ollama](https://ollama.com/download)** — lets you run a small LLM locally for free
- A working internet connection (for trivia questions and the first model pull)

Once Ollama is installed, pull the model the game uses:

```bash
ollama pull llama3.2:1b
```

> You only need to do this once. After the pull, everything runs fully offline.

---

## Step 1 — Understand the game state

Before looking at any nodes or logic, start here: the **game state**.

`main.py` builds a small **state machine** using [LangGraph](https://langchain-ai.github.io/langgraph/). Think of it as a graph where each node is a step in the game, and a single shared object flows between them carrying all the information the game needs.

That object is `TriviaState`:

```python
class TriviaState(TypedDict):
    score: int             # how many questions you've gotten right
    total_questions: int   # how many you've been asked
    question: str          # the current question text
    choices: list          # the shuffled multiple-choice options
    correct_answer: str    # the right answer (ground truth)
    user_answer: str       # what you selected
    difficulty: str        # easy / medium / hard
    game_over: bool        # flag that ends the loop
```

Using a `TypedDict` gives every field a clear type, making it easy to see exactly what data each node can read and update.

The LLM is created once at module load:

```python
llm = ChatOllama(model="llama3.2:1b", temperature=0.4)
```

This points at Ollama running locally. The low `temperature` keeps the judge's verdict focused and consistent.

---

## Step 2 — Walk through each node

Each node is a plain Python function: it receives the current `TriviaState` and returns a dictionary of fields to update. Here's what each one does:

### `setup_node` — Welcome the player

Greets the player and asks them to choose a difficulty (`easy` / `medium` / `hard`). Invalid input defaults to `easy`. This node initialises `score`, `total_questions`, and `game_over`.

### `fetch_question_node` — Get a trivia question

Builds a URL for the [Open Trivia DB API](https://opentdb.com/) using the chosen difficulty, fetches one multiple-choice question, and cleans it up:

- `html.unescape(...)` decodes HTML entities (e.g. `&quot;` → `"`).
- Correct and incorrect answers are combined into a single `choices` list and `random.shuffle`-d so the right answer isn't always in the same position.

### `present_question_node` — Show the question and read input

Prints the question and its numbered choices, then waits for input. Typing `q` quits the game (`game_over = True`). Otherwise it records the selected answer and increments `total_questions`. Bad input is treated as a wrong answer.

### `judge_node` — Let the LLM judge

Builds a prompt that contains the question, the correct answer (as ground truth), and the player's answer, then calls the LLM to give a friendly verdict in 1–2 sentences. The score is updated, the running total is printed, and the player is asked whether to continue — anything other than `y` sets `game_over = True`.

---

## Step 3 — Understand the routing

After certain nodes, the graph needs to decide where to go next. Two small **routing functions** handle this:

- **`route_after_present_question`** — if you quit, go to `END`; otherwise proceed to `judge`.
- **`route_after_judge`** — if the game is over, go to `END`; otherwise loop back to `fetch_question` for another round.

These functions return a string label that the graph maps to the next node.

---

## Step 4 — See how the graph is wired together

`build_graph()` connects all the nodes and routing functions:

```text
setup → fetch_question → present_question ─┬─(quit)──────────────→ END
                                           └─(answer)→ judge ─┬─(continue)→ fetch_question
                                                              └─(stop)────→ END
```

Here's how each part of the graph is built:

| Method | Purpose |
|--------|---------|
| `graph.add_node(...)` | Register a node by name |
| `graph.add_edge(...)` | Add a fixed transition between nodes |
| `graph.add_conditional_edges(...)` | Map routing labels to the next node |
| `graph.compile()` | Produce the runnable app |

The `__main__` block creates an empty initial state, calls `app.invoke(...)` to run the entire loop, and prints the final score when the game ends.

---

## Step 5 — Run the game

You're ready! Start the game with:

```bash
uv run main.py
```

You'll be prompted to choose a difficulty, then served questions one at a time. Type the number of your answer and press Enter. Type `q` at any time to quit. After each answer, choose whether to keep playing.

---

## Troubleshooting

**`ConnectionError` to `localhost:11434`**
Ollama isn't running. Start it with:
```bash
ollama serve
```
Or restart the Ollama desktop app.

**Model not found / `model "llama3.2:1b" not available`**
The model hasn't been pulled yet. Run:
```bash
ollama pull llama3.2:1b
```

**Unexpected question encoding (e.g. `&amp;`)**
This shouldn't happen — the code uses `html.unescape` — but if it does, check your internet connection and retry.
