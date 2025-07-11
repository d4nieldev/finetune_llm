## The QPL Language

QPL is a formalism used to describe data retrieval operations over an SQL schema in a modular manner. A QPL plan is a sequence of instructions for querying tabular data to answer a natural language question.

The top-level operators in the QPL language are:
    * **Scan** - Scan all rows in a table with optional filtering predicate (no decomposition needed - the question is atomic)
    * **Aggregate** - Aggregate a stream of tuples using a grouping criterion into a stream of groups (1 sub-question)
    * **Filter** - Remove tuples from a stream that do not match a predicate (1 sub-question)
    * **Sort** - Sort a stream according to a sorting expression (1 sub-question)
    * **TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 sub-question)
    * **Join** - Perform a logical join operation between two streams based on a join condition (2 sub-questions)
    * **Except** - Compute the set difference between two streams of tuples (2 sub-questions)
    * **Intersect** - Compute the set intersection between two streams of tuples (2 sub-questions)
    * **Union** - Compute the set union between two streams of tuples (2 sub-questions)

Every line of QPL is the execution of an operator on its inputs. In case where the operator is Scan, the input is the table being scanned. The output of a QPL plan is the last line of that QPL code.


## The User's Task

The user's task is to decompose the natural language question into an appropriate QPL operator and sub-questions, that when combined have the same semantic meaning as the main question.

**Input**
    * Database schema
    * Question in natural language (about this database)
**Output**
    * QPL operator
    * Sub-questions - depending on the operator - 0-2 simpler sub-questions - that serve as input to the operator.


## Your Task

Your task is to provide reasoning that mimics the user thought process from the point they received the natural language question and database schema to the point they chose the operator and sub-questions.

**Input**
    * Database schema
    * Question in natural language
    * QPL operator
    * Sub-questions
    * Full QPL code to answer the question
**Output**
    * Reasoning on how to get from the database schema and question to the decomopsition (operator and sub-questions)

**Note About the Full QPL code**

You can consult the full QPL code to better understand the user's choice, however, the user wrote the full QPL code only at a later stage that require more decomposing so **do not refer to it or use it to explain the next stages of the implementation**, the full QPL code can help mainly for resolving ambiguities or explaining why choosing the exact operator when another operator might seem suitable as well. Notice that the last line of the QPL is ALWAYS the line that puts it all together and answers the question - so this might be  "closer" to what the user thought while decomposing the question than the other lines in the QPL code.


### Reasoning Steps

The user follows this logical progression when decomposing the question:

#### Determine the Operator

Start by interpreting the question’s main intent. 
Think about **what is the form** of the question -  generally, the unary operators have the logic: "First (sub-question), then (apply operator)", and the binary operators have the logic: "Find (sub-question-1), Find (sub-question-2) then combine by (applying operator)"
Find out what would be the **final operation** applied in order to answer the question - look for action words or logical structure that hint at the underlying data operation. Use the following cues to guide your selection of the appropriate QPL operator:

* **Scan**:
  Look for *simple fact-finding* or *attribute lookup* questions that can be answered directly from a single table without computation or combination.
  *Cues: “List all…”, “Show me…”, “What is the value of…”*
* **Aggregate**:
  Indicates *grouping* or *summarization* over a set of items, often using counts, sums, averages, or other metrics.
  *Cues: “How many…”, “Total number of…”, “Average… per…”, “Group by…”*
* **Filter**:
  Suggests applying *conditions* to remove irrelevant rows.
  *Cues: “Only those that…”, “Where…”, “Which… satisfy…”, “Exclude…”, “With condition…”*
* **Sort**:
  Implies *ordering* the results based on a column but without restricting how many items are returned.
  *Cues: “In order of…”, “Sort by…”, “Ranked according to…” (without specifying top-K)*
* **TopSort**:
  Involves both *sorting* and selecting a *top-K subset*.
  *Cues: “Top 5 by…”, “Most recent…”, “Biggest…”, “Best ranked…”, “Highest rated…”*
* **Join**:
  Needed when information must be *combined across multiple tables*, often via matching keys or relationships.
  *Cues: “Who (in one table) did what (in another)…”, “For each…, find their…”*
* **Except**:
  Indicates *exclusion* of one group from another.
  *Cues: “Who did not…”, “All except…”, “Exclude those who…”*
* **Intersect**:
  Suggests selecting entities that *appear in both sets*.
  *Cues: “Who both… and…”, “Entities that meet multiple criteria from different sources”*
* **Union**:
  Combines entities from *two distinct groups*.
  *Cues: “Who either… or…”, “Combine the results of…”, “All entities from both…”*

#### Formulate Sub-Questions

**IF THE OPERATION IS SCAN - SKIP THIS REASONING STEP AND MOVE DIRECTLY TO JUSTIFYING THE DECOMPOSITION**

Break down the question according to the chosen operator. First, discuss on what each sub question need to achieve in high level. Then, define 1 or 2 immediate sub-questions (as required by the operator’s arity), ensuring that **applying the selected operator to the sub-questions will reconstruct the logic of the original question**. Clearly describe how each sub-question aligns with parts of the original question and with the schema.

#### Justify the Decomposition

Explicitly state the selected operator and sub-questions (if there are) verbatim. Summarize why the chosen operator and the sub-questions **combined** accurately reflect the original question’s requirements. Reference the schema to support your reasoning, ensuring that the decomposition is complete and logically sound.


### General guidelines:

* Stay on high-level decomposition of the current question, and DO NOT dive into the implementation especially not revealing the QPL code or the next steps.
* Explain only the current decomposition step - do not explain the further decompositions needed or how to retrieve the information for the sub-questions.
* In any of the reasoning steps, do not get dragged to explaining the full plan for the question, stay focused solely on finding out how to decompose it.
* Avoid unnecessary explanations - DO NOT repeat yourself.
* Focus on brevity while maintining a logical, step-by-step explanation without skipping any logical step.
* Output your answer in markdown format where each reasoning step has its own title and body.