import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = r"C:\Users\dhruv\Desktop\explainable_robust_absa\results\transformer\proposed_robust_model\best_model"


# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully!")
print()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(aspect, sentence):

    # IMPORTANT:
    # Your training code used:
    #
    # tokenizer(aspect, sentence)
    #
    # Therefore we must use the same format here.

    inputs = tokenizer(
        aspect,
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        outputs = model(**inputs)

    # Raw model scores
    logits = outputs.logits

    # Convert scores to probabilities
    probabilities = torch.softmax(
        logits,
        dim=-1
    )[0]

    # Get highest probability class
    predicted_id = torch.argmax(
        probabilities
    ).item()

    predicted_label = LABEL_NAMES[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item()

    return predicted_label, confidence, probabilities


# ============================================================
# INTERACTIVE TEST
# ============================================================

print("=" * 60)
print("PROPOSED ROBUST DISTILBERT - ABSA TEST")
print("=" * 60)

print()
print("Enter 'exit' to quit.")
print()


while True:

    aspect = input("Aspect: ").strip()

    if aspect.lower() == "exit":
        break

    sentence = input("Sentence: ").strip()

    if sentence.lower() == "exit":
        break

    if not aspect or not sentence:

        print(
            "Please enter both aspect and sentence."
        )

        continue

    label, confidence, probabilities = (
        predict_sentiment(
            aspect,
            sentence
        )
    )

    print()
    print("-" * 60)

    print(
        f"Aspect     : {aspect}"
    )

    print(
        f"Sentence   : {sentence}"
    )

    print(
        f"Prediction : {label}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print()
    print("Probabilities:")

    for class_id, class_name in LABEL_NAMES.items():

        probability = probabilities[
            class_id
        ].item()

        print(
            f"  {class_name:<10}: "
            f"{probability * 100:.2f}%"
        )

    print("-" * 60)
    print()
