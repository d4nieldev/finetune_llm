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

Below is the formal specification for each operation in valid QPL:

<qpl> ::= <line>+
<line> ::= #<integer> = <operator>
<operator> ::= <scan> | <aggregate> | <filter> | <sort> | <topsort> | <join> | <except> | <intersect> | <union>

-- Leaf operator
<scan> ::= Scan Table [ <table-name> ] <pred>? <distinct>? <output-non-qualif>

-- Unary operators
<aggregate> ::= Aggregate [ <input> ] <group-by>? <output-agg>
<filter> ::= Filter [ <input> ] <pred> <distinct>? <output-non-qualif>
<sort> ::= Sort [ <input> ] <order-by> <output-non-qualif>
<topsort> ::= TopSort [ <input> ] Rows [ <number> ] <order-by> <withTies>? <output-non-qualif>

-- Binary operators
<join> ::= Join [ <input> , <input> ] <pred>? <distinct>? <output-qualif>
<except> ::= Except [ <input> , <input> ] <pred> <output-qualif>
<intersect> ::= Intersect [ <input> , <input> ] <pred> <output-qualif>
<union> ::= Union [ <input> , <input> ] <output-qualif>

<group-by> ::= GroupBy [ <column-name> (, <column-name>)* ]
<order-by> ::= OrderBy [ <column-name> <direction> (, <column-name> <direction>)* ]
<withTies> ::= WithTies [ true | false ]
<direction> ::= ASC | DESC
<pred> ::= Predicate [ <comparison> (AND | OR <comparison)* ]
<distinct> ::= Distinct [ true | false ]
<output-non-qualif> ::= Output [ <column-name> (, <column-name>)* ]
<output-agg> ::= Output [ <agg-column-name> (, <agg-column-name>)* ]
<output-qualif> ::= Output [ <qualif-column-name> (, <qualif-column-name>)* ]
<agg-column-name> ::= countstar | <agg-func>(<column-name>) | <agg-func>(DISTINCT <column-name>)
<agg-func> ::= COUNT | SUM | AVG | MIN | MAX
<qualif-column-name> ::= #<number>.<column-name>

Generally every QPL line has 4 parts:
1. Operator
2. Input Streams
3. Options
4. Output Columns


## The User's Task

The user's task is to add one line to a given prefix QPL, such that the added line concatenated to the prefix QPL will form a QPL query that will answer a given natural language question.

**Input**
    * Database schema
    * Question in natural language
    * Prefix QPL lines such that each line ends with a comment that describes the question it answers
    * The QPL operator of the final line to add (part 1)
    * The identifiers of the input streams (part 2) - only if the operator is not "Scan"
**Output**
    * Completion of the operator and inputs to form the final line


## Your Task

Your task is to provide reasoning that mimics the user thought process from the point they received the database schema, natural language question, prefix QPL, and start of the final QPL line to add to the point they finished writing the QPL line.

**Input**
    * Database schema
    * Question in natural language
    * Prefix QPL lines such that each line ends with a comment that describes the question being addresseed
    * The final final QPL line to add (the part given to the user is highlighted between "*" and "*" and the rest is to be predicted)
**Output**
    * Reasoning on how to get from the database schema, natural language question, prefix QPL, and start of the final QPL line to add to the final QPL line (completion of given start)

**Note About the Given Final Line To Add**

You are given the desired user's response, but the user does not have this answer when first given the question. Therefore, you must not refer to it as given, you must detail every step until constructing the final line without revealing what's ahead.


### Reasoning Steps

The user builds the final QPL line step by step following this logical progression:

#### Understanding the Input Stream

First, the user must understand what is the input to the final row. There are 2 options:

* If the operator is "Scan", meaning the user is not given the input table to scan, so this part should examine all the given tables in the schema based on their names and columns (especially primary keys) and decide which table contains the columns requested by the given question.
* Otherwise, the input streams are given, and the user must analyze their columns and intent based on the given prefix QPL code and the corresponding comments in it to better understand it before processing it in the following reasoning steps.

In the end of this step, you must provide parts 1-2 of the final QPL line (part 1 is always given, part 2 is given only if the operator is not "Scan")

#### Operator-Specific Options

Each operator have diffrent options that define how the input stream is being processed. Depending on the operator and the question, you should reason on every possible option for that operator syntax and determine whether the option is relevant or not - and if it is relevant, how to use it to **process the input streams**. Here are guiding questions for each operator and option, as illustrated by the formal language specification:

* **Scan**: 
    - Is filtering by a *predicate* required by the question? By which columns and values?
    - Is returning *distinct* values was asked for in the question, or duplicates are fine?
* **Aggregate**:
    - Is there a need to *group by* specific columns, or all of the rows need to be aggregated?
* **Filter**:
    - How to write the *predicate* required by the question? What are the columns and values that must be involved?
    - Is returning *distinct* values was asked for in the question, or duplicates are fine?
* **Sort**:
    - By which *column* the input stream must be sorted?
    - Should the data be sorted in *ascending* or *descending* order?
* **TopSort**:
    - *How many* rows to return? Is it specified in the question or implied?
	- Which columns determine the sorting? What should the *order* (ascending/descending) be?
	- Should *WithTies* be enabled (rows with equal values in the sorting column be included beyond the number of rows to return)?
* **Join**:
	- Does the question specify a *join condition* (e.g., matching IDs)? Which columns from each input stream should be used for the join? Explain **why these columns are suitable for joining**, possible reasons are: a foreign key–primary key relationship, identical column names, or other relevant correspondences.
	- Is returning *distinct* values was asked for in the question, or duplicates are fine?
* **Except** (note: the first input stream is considered the subtrahend, the second is the minuend):
	- What is the *except matching condition* (if true for a row, this row will be included in the output)?
* **Intersect**:
	- What is the *intersect matching condition* (if true for a row, this row will be included in the output)?

Note: Specifically for the "Union" operator, there are no special options - it just combines the two input streams, so this reasoning step must be skipped if the operator for the final line is a "Union".

In the end of this step, you must provide parts 1-3 of the final QPL line (everything without the "Output" part)

#### Output Column Selection

Finally, the user must decide which columns to include in the output. The output columns can only be selected from the output columns of the input streams (for "Scan" - where the input stream is a table, every column in that table can appear in the output) or be a literal value, and must match exactly what the user asked. 
For "Aggregate" lines - every column in the output must either be included in the "GroupBy" option, or aggregated by an aggregation function (MIN/MAX/SUM/COUNT/...) - use aliases for readability.
For "Join" lines - discuss what relevant information each input stream brings to the table and which columns to extract from which input stream to fully address the given natural language question.

In the end of this step, you must provide all parts (1-4) of the final QPL line.


### General Guidelines

* Conclude every intermediate step with "The final QPL line so far is: ..." and the final step should end with "The final QPL line is: ..."
* DO NOT refer to the QPL line general part numbers - this is just a clarification for you to understand the structure of a QPL line, the user is already familiar with it.
* DO NOT make assumptions such as what is likely and what is not about the input streams, you have no idea how the data looks like and it can sometimes be confusing. Avoid using words like "likely", "unlikely", "probably", ...
* The final QPL line MUST match exactly the provided target final QPL line. In some cases, this line may include additional columns that were necessary for intermediate processing, even if they are not explicitly requested in the original question. These columns should still be included, but make sure to note that their inclusion is for clarity or to aid in understanding the output, rather than being a direct requirement of the question.
* Avoid unnecessary explanations - DO NOT repeat yourself.
* Focus on brevity while maintining a logical, step-by-step explanation without skipping any logical step.
* Output your answer in markdown format where each reasoning step has its own title and body.
