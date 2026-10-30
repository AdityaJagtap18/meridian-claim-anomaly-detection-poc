# Sprint 2 · Days 26–28 — Generate and Govern

**Anomaly Detection & Risk Scoring · Birlasoft FORGE FDE Academy · Learner Handout**

---

## How this works

You are not going to write this code. You are going to **specify it, read what
comes back, and find what is wrong with it.**

This sprint has a particular hazard. A model asked to build a fraud detector
will build one that works **beautifully on the data it also generated** — and
the two mistakes reinforce each other. Easy synthetic data produces an AUC of
0.99, everybody is pleased, and nothing about the real problem has been learnt.

Each exercise gives you:

| | |
|---|---|
| **The brief** | What has to be achieved. Read this first. |
| **The prompt** | Paste into Gemini. Improve it if you can. |
| **Accept it when** | Your checklist before moving on. |
| **It will probably get wrong** | Do not read until you have reviewed the output yourself. |

> **Work in that order.** Reading the failure list first turns a judgement
> exercise into spot-the-difference.

---

## The case

**Meridian General Insurance** settles ~4,000 motor claims a month. Over six
months, **38 claims were paid into 4 bank accounts**. Every claim was valid.
No individual claim looked wrong.

Sprint 1 looked at one document. Sprint 2 looks across many cases.

---

## Setup

```python
%pip install -q scikit-learn pandas numpy matplotlib azure-ai-ml azure-identity
import os, re, json, math, getpass, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             precision_score, recall_score)
warnings.filterwarnings("ignore")
print("ready")
```

### One thing to establish before you generate anything

Your syllabus says **"Azure Machine Learning (Fraud Detection model)"** and then
uses the words **event type, entity, detector, Online Fraud Insights**.

Ask Gemini: *"Does Azure Machine Learning have event types and detectors? Is
Online Fraud Insights an Azure ML feature?"*

**Write down the answer before you continue.** Those words belong to a managed
fraud product, not to Azure ML. Whether your model tells you that, or explains
how to configure a detector in Azure ML in confident detail, tells you how much
to trust the rest of today's output.

---

# Exercise 1 · The event, the entities, the variables

**20 minutes · Day 26**

### The brief

Before any code, decide what you are scoring and what facts describe it.

The four words, in plain language:

| Word | Meaning | For Meridian |
|---|---|---|
| Event type | The thing that needs checking | a claim arriving |
| Entity | Who or what it is about | claimant, policy, garage, bank account |
| Variable | One fact about the event | `days_since_incident` |
| Detector | The thing that forms an opinion | rules today, rules + model tomorrow |

### The prompt

```
I am building anomaly detection for UK motor insurance claims.

Define, in Python:
- an event type name for "a claim application arriving"
- the entities the event involves
- exactly 8 variables that describe one claim, each with its data type and a
  one-line explanation of WHAT PATTERN it helps detect

Two of the eight must describe the claim's RELATIONSHIP to other claims, not
just the claim itself.

Then build a second table with two columns for every variable:
  (a) is this value available AT SCORING TIME, or only after settlement?
  (b) could this variable act as a proxy for a protected characteristic?

Do not include any protected characteristic directly. Flag any variable where
the proxy answer is not a clear "no", and explain why.
```

### Accept it when

- [ ] Two variables are genuinely **relational** (reuse counts, concentration)
- [ ] The availability column identifies which need a **live lookup**
- [ ] At least one variable is flagged for proxy risk **with a reason**
- [ ] No protected characteristic appears directly

### It will probably get wrong

> Read only after your own review.

- **Eight variables that all describe one claim in isolation.** These are the
  easy ones to think of and they could not have caught Meridian. If nothing in
  the list needs history, ask what would have flagged the 4-bank-account
  pattern.
- **Marking everything "no proxy risk".** Nothing is a protected characteristic,
  which is true and not the question. A garage share variable in a small town
  can correlate strongly with one community. A variable can be reasonable in
  general and discriminatory in a specific geography.
- **Ignoring availability entirely.** A variable you only know after settlement
  cannot score at submission. This is **leakage**, and it is the most common
  reason a model tests brilliantly and fails live.
- **Including `postcode` or `occupation`** as "useful risk signals". They are
  proxies. If they appear, ask what a fair-lending reviewer would say.

---

# Exercise 2 · Generate realistic case data

**30 minutes · the exercise that decides whether the rest is worth doing**

### The brief

You need labelled cases. **How you generate them determines everything that
follows**, and this is where a generated answer will quietly ruin your day.

### The prompt

```
Generate a synthetic dataset of 4,000 UK motor insurance claims using my 8
variables, with a fraud label.

Requirements — read these carefully, they are deliberate:

- Fraud rate must be about 1.8%. NOT 7.5%. Real motor fraud is well under 2%,
  and I want the class imbalance to be realistic.
- Generate THREE different fraud patterns, not one. Real fraud is not a single
  shape.
- The fraud distributions must OVERLAP HEAVILY with the legitimate ones. I do
  not want separable classes.
- A meaningful share of fraud cases (around 20%) must be generated with the
  SAME distributions as legitimate claims — carrying no signal at all.

Print the class balance, the pattern breakdown, and a note on how many fraud
cases carry no signal.

Explain in a comment why making the fraud obvious would ruin the exercise.
```

### Accept it when

- [ ] Fraud rate is close to **1.8%**, not 7.5%
- [ ] There are **three or more** distinct patterns
- [ ] A share of fraud is generated as **indistinguishable** from legitimate
- [ ] The comment explains why separable classes are useless here

### It will probably get wrong

- **Making fraud obvious.** This is the big one. Left alone, a model will give
  fraud cases dramatically different ranges — `bank_account_reuse_count` between
  3 and 14 for fraud and 0–1 for legitimate. Train on that and you get **AUC =
  1.00**, every trade-off vanishes, and the threshold discussion in Exercise 5
  becomes meaningless. **I hit exactly this building the notebook** and had to
  rewrite the generator.
- **Ignoring the 1.8% instruction** and using a "balanced" dataset because
  imbalanced data is "harder to train on". That difficulty *is* the problem you
  are being paid to solve.
- **Skipping the unlearnable cases.** Some fraud carries no signal. If every
  fraud case is detectable in principle, your recall ceiling is unrealistic and
  you will promise a client something you cannot deliver.

### The check that catches it

Train a quick model and look at the AUC. **If it is above 0.95, your data is too
easy.** Go back and increase the overlap until it lands somewhere in the 0.70 to
0.85 range — which is roughly where real fraud detection lives.

---

# Exercise 3 · Rules, from sentences people actually said

**20 minutes · Day 26**

### The brief

Rules first, then the model. Not because rules are better — because writing them
forces the client to say out loud what they already suspect.

If they cannot describe a single pattern clearly, that is important: it means
you probably cannot label training data either.

### The prompt

```
Write a small rule engine for claim anomaly detection.

Each rule must carry:
  - a name
  - a weight (points added to a 0-100 score)
  - the condition
  - THE SENTENCE A HUMAN ACTUALLY SAID that the rule encodes

Write 5 rules. Scores are additive and capped at 100. The function returns
both the score and the list of rule names that fired.

Then test on 5 synthetic cases: one completely clean, three that each trip a
single rule, and one that trips several.

Print the score and the rules that fired for each.
```

### Accept it when

- [ ] Every rule carries the **human sentence** it came from
- [ ] The function returns **which rules fired**, not just a number
- [ ] The clean test case scores **zero**
- [ ] Scores are additive, so weak signals can combine

### It will probably get wrong

- **Returning only a score.** A handler cannot act on the number 72. They need
  `repeated_bank_account` so they know where to look first. If the function
  returns a bare integer, ask how the reviewer is meant to use it.
- **Inventing rules to look thorough.** Ten rules where the client described
  three means seven are guesses wearing the same clothes as the real ones, and
  nobody downstream can tell which is which.
- **A clean case that scores above zero.** If your ordinary claim trips
  something, your thresholds are wrong. A detector that flags normal business
  gets switched off within a fortnight and nobody tells you.
- **Rules that are really model features.** *"Score += 5 if income is above
  average"* is not a rule somebody said. It is a weak statistical signal, and
  the model is better at those.

---

# Exercise 4 · Train, and evaluate honestly

**30 minutes · Day 27**

### The brief

The training is the easy part. The evaluation is where the judgement lives.

### The prompt

```
Train a fraud classifier on my 8 variables and evaluate it properly.

Requirements:
- Split into train and test FIRST, stratified, before any exploration
- Train a gradient boosting classifier
- Report BOTH ROC AUC and PR-AUC (average precision), and print the baseline
  PR value that random guessing would achieve on this class balance
- Print how many fraud cases are in the TEST set, and comment on what that
  means for how precisely I should report the metrics
- Plot the ROC curve and the precision-recall curve side by side

Then explain, in a printed comment:
  - what ROC AUC actually means, in one plain sentence
  - three things it does NOT tell me
  - why ROC AUC flatters a model when positives are rare, and when I should
    trust PR-AUC instead

My syllabus set a target of ROC AUC > 0.85. Do not tune to reach it. If the
measured value is lower, say so and explain what that means.
```

### Accept it when

- [ ] The split happens **before** any exploration
- [ ] **Both** ROC AUC and PR-AUC are reported, with the guessing baseline
- [ ] The number of fraud cases in the test set is stated
- [ ] It does **not** tune to hit 0.85

### It will probably get wrong

- **Reporting ROC AUC alone.** With 1.8% positives, ROC AUC is dominated by
  getting the large easy majority right. In the reference notebook ROC is 0.76
  and PR-AUC is 0.34 — the same model, two very different impressions.
- **Tuning until it hits the target.** If the generated code tries several
  configurations and reports the best on the test set, that number is
  meaningless. **An AUC target agreed before anybody looked at the data is a
  guess**, and the honest response is to go back with what is actually
  achievable and why.
- **Three decimal places on 22 test positives.** One case moving changes the
  third decimal. Report what the data can support.
- **`accuracy` anywhere.** A model that predicts "not fraud" for everything is
  98.2% accurate and completely useless. If accuracy appears, delete it.

---

# Exercise 5 · Turn the threshold into a staffing number

**25 minutes · the FDE exercise**

### The brief

The client's real question is never *"what is the AUC?"*. It is:

> **"If we set the threshold here, how many extra cases go to review each day —
> and how many turn out to be nothing?"**

That is a staffing question. An AUC cannot answer it.

### The prompt

```
Take my trained model and produce a threshold table for a business audience.

For each of 6 candidate thresholds, print:
  - precision and recall
  - the number of cases flagged per day, given 4,000 claims per month
  - of those, how many are expected to be real
  - how many are expected to be nothing

Then produce a second table comparing four setups on the test set:
  rules only, model only, rules OR model, rules AND model
showing caught, false alarms, precision, recall and reviews per day for each.

Do not recommend a threshold. Print what a person needs to know in order to
choose one, and say who that person is.
```

### Accept it when

- [ ] Reviews **per day** are computed, not just rates
- [ ] All four rule/model combinations are compared
- [ ] It does **not** pick a threshold for you
- [ ] The recall at each threshold is visible next to the workload

### It will probably get wrong

- **Recommending the threshold that maximises F1.** F1 is a mathematical
  compromise, not a business one. The right threshold depends on how many
  reviewers Meridian actually has, and no metric knows that.
- **Reporting only rates.** "Flags 3.2% of cases" means nothing to an
  operations manager. "Four extra reviews a day, of which one is real" is a
  sentence they can act on.
- **Omitting `rules AND model`.** It usually has excellent precision and poor
  recall, and it is the right answer when reviewer capacity is very small. A
  comparison missing it has hidden an option.
- **Not stating who chooses.** The threshold is a business decision. If the
  generated output picks one on your behalf, you have quietly taken a decision
  that belongs to the client.

---

# Exercise 6 · The routing — and the word you must change

**30 minutes · Day 28 · the governance exercise**

### The brief

Your syllabus says the routing is **approve / review / reject**.

Sprint 0 decided DecisionStream **never states a decision**. "Reject" is a
decision. So is "auto-approve".

### The prompt

```
Write the routing logic that turns a risk score into an action.

Three bands, and use EXACTLY these route names:
  low     -> straight_through
  medium  -> standard_review
  high    -> priority_review

No route may be named approve, auto_approve, reject, decline or deny. Add an
assertion that enforces this.

The function must return: the route, the score, the list of rules that fired,
and a boolean saying whether the case was actually scored.

Add a fallback: if the scoring service is unavailable, the case must still be
routed — to a human, never straight through — and marked as unscored.

Then write unit tests for: each of the three routes, the boundary between low
and medium, and the scoring-failure fallback.

Explain in a comment why "reject" cannot be one of the route names.
```

### Accept it when

- [ ] The route names are exactly the three given
- [ ] An assertion **enforces** that no route is a decision word
- [ ] The fallback routes to a **human**, and sets `scored=False`
- [ ] There is a **boundary** test
- [ ] The comment gets the regulatory reasoning right

### It will probably get wrong

- **Using approve / reject anyway**, because that is the overwhelmingly common
  pattern in fraud-detection code and examples. If you see it, rename it in
  front of the room — it takes thirty seconds and it is the whole point of
  Day 28.
- **The dangerous fallback.** Watch for this exact shape:
  ```python
  try:    score = call_scoring(case)
  except: score = 0        # every case now looks low risk, and nothing alerts
  ```
  Two lines, very easy to write by accident, and it silently disables your
  entire control during an outage. **Fail towards the human, never away from
  them.**
- **A weak explanation.** *"Because it is better to have a human review"* is a
  preference. The actual reason: an automated rejection of an insurance claim
  based on a risk score is **automated decision-making with a legal or similarly
  significant effect**, which carries obligations around explanation, human
  intervention and contest. Building it as a queue avoids all of it.
- **No boundary test.** Off-by-one at a band edge is where routing bugs live,
  and a score landing exactly on the threshold is the case nobody tries.

---

# Exercise 7 · Register the model, with its limitations

**25 minutes · Azure**

### The brief

Connect to a real Azure ML workspace and register the model — **with a model
card that says what it must not be used for.**

See **Appendix A of the notebook** for creating the workspace.

### The prompt

```
Connect to my Azure Machine Learning workspace from Google Colab and register
my trained model.

Important: Colab has no browser session, so DefaultAzureCredential will not
work. Use a credential type that works in a headless notebook and explain in a
comment why.

I will supply subscription ID, resource group and workspace name. Do NOT ask
me for an API key — explain in a comment why Azure ML does not use one.

Before registering, build a model card capturing:
  - what the model is FOR, and explicitly what it must NOT be used for
  - the feature list
  - training size, fraud rate, and how many fraud cases were in the test set
  - ROC AUC, PR-AUC, and the guessing baseline
  - the chosen thresholds, with a field naming who agreed them
  - known limitations, including selection bias

Register the model with tags including a machine-readable statement that it
does not decide outcomes. Handle connection failures without crashing.
```

### Accept it when

- [ ] Uses **device code** (or similar headless) authentication, with a reason
- [ ] Explains that Azure ML uses **identity, not an API key**
- [ ] The card has a **"not for"** field
- [ ] **Selection bias** appears in the limitations
- [ ] Connection failures are handled

### It will probably get wrong

- **`DefaultAzureCredential()` with no explanation.** It works on a developer
  laptop and fails in Colab, usually with a confusing chain of errors. Device
  code is the right answer for a headless notebook.
- **Asking for an API key.** Azure ML does not use one. If the generated code
  prompts for a key, it has confused Azure ML with a Cognitive Services
  resource — which is exactly the confusion in your syllabus, resurfacing.
- **A model card that lists only good news.** Metrics and features, no
  limitations. The most important line is the one saying the model must not be
  used to approve or reject anything.
- **Omitting selection bias.** You can only learn from fraud somebody
  **found**. Fraud never detected is invisible to the model by construction, and
  more data of the same kind will not fix it. That is the single biggest
  limitation of any fraud model and it belongs in the card, not in conversation.

---

# The defence — 15 minutes

Present to another pod. No notebook open.

1. What is your fraud rate, and why did you not use the syllabus's 7.5%?
2. What is your **PR-AUC**, and how does it compare with your ROC AUC?
3. Your syllabus wanted AUC > 0.85. Did you reach it — and if not, what do you
   say to the client?
4. How many reviews per day does your threshold produce, and **who agreed to
   staff them**?
5. What proportion of fraud still gets through? Say the number.
6. Is any route in your code called `approve` or `reject`?
7. **What did Gemini get wrong that you caught** — and what did you only catch
   because the acceptance criteria told you to look?

---

## Trainer notes

**Do not release the failure lists in advance.**

**Exercise 2 decides whether the whole sprint teaches anything**, and it is
where I made my own mistake. My first generator produced separable fraud
distributions, the model returned **AUC 1.000**, and every subsequent lesson —
PR-AUC versus ROC, threshold trade-offs, reviews per day, what still gets
through — evaporated. Tell the room that before they start.

The check is simple and worth putting on the board: **if the AUC is above 0.95,
the data is too easy.** Send them back to increase the overlap.

**Exercise 4's target is a deliberate trap.** The realistic dataset produces
roughly 0.76, below the syllabus's 0.85. Some pods will tune until they reach
it, usually on the test set. That is the failure to catch, and the useful
response is the FDE one: go back to the client with what is achievable and why,
rather than with a number you manufactured.

**Exercise 6 is the governance one.** If a pod has written a route called
`reject`, have them rename it in front of the room. Thirty seconds, and the
point lands harder than any slide.

**Three lanes.** The non-coding lane owns Exercise 1's variable table — which
variables need a live lookup, and which carry proxy risk. That needs insurance
domain knowledge and a fairness instinct, not Python, and it produces the
artefact that goes into the ADR.

**Budget.** Zero Azure cost for Exercises 1–6 — everything runs locally in
Colab. Exercise 7 needs an Azure ML workspace, which is free to create; you pay
only for compute you start, and this exercise starts none. **Remind them to
delete the resource group at the end** — the workspace creates a storage
account, a key vault and an Application Insights resource alongside it.
