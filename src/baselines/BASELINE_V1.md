# Classical Baseline V1

Dataset:
MAMS-ATSA

Task:
Aspect-Term Sentiment Classification

Input:
[ASPECT] aspect [CONTEXT] sentence

Train:
11,186 aspect instances

Validation:
1,332 aspect instances

Test:
1,336 aspect instances

Random seed:
42

Baseline:
Majority Class

Test Accuracy:
45.43%

Test Macro-F1:
20.83%

Classical Model:
TF-IDF + Linear SVM

TF-IDF:
Word n-grams: 1-2
Character n-grams: 3-5
sublinear_tf=True

SVM:
LinearSVC
C=1.0
class_weight=None

Test Accuracy:
66.17%

Test Macro-F1:
64.28%

Test Weighted-F1:
65.75%

Test per-class F1:
Negative: 63.76%
Neutral: 73.58%
Positive: 55.50%

Primary observed weakness:
Positive-as-neutral confusion.

Positive recall:
51.75%

Positive → Neutral errors:
128

Status:
FROZEN BASELINE V1