"""
11_inference_demo.py
======================
Interactive demo: type in any Urdu sentence + aspect, get a live
sentiment prediction from your trained XLM-R model. This is what you
can run live during your defense to show the model actually working.

REQUIRES: 10_full_data_and_save.py must have been run first (this is
what creates saved_models/xlmr_full_data/, the model this script loads).

HOW TO RUN (in Colab):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/11_inference_demo.py"

Two ways to use it once it's running:
  1. It first runs through a few BUILT-IN example sentences (one per
     domain) automatically, so you have a working demo even without
     typing anything -- good for a live defense where you don't want
     to fumble typing Urdu text on the spot.
  2. Then it drops into an interactive loop where you can type your
     own sentence + aspect and see the prediction, with the model's
     confidence for all 3 classes shown (not just the top pick) --
     type 'quit' to exit the loop.
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import CFG, set_global_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A few ready-made examples, one per domain, so the demo works
# immediately even before you type anything yourself.
BUILT_IN_EXAMPLES = [
    {"sentence": "یہ کھانا بہت مزیدار تھا اور جلدی مل گیا۔", "aspect": "کھانا", "domain": "food"},
    {"sentence": "پاکستان نے آج کا میچ شاندار انداز میں جیت لیا۔", "aspect": "پاکستان", "domain": "sports"},
    {"sentence": "وزیراعظم کی تقریر بہت مایوس کن تھی۔", "aspect": "وزیراعظم", "domain": "news_politics"},
]


def load_model():
    model_path = os.path.join(CFG.MODELS_DIR, "xlmr_full_data")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at {model_path} -- run "
            "10_full_data_and_save.py first to train and save it."
        )
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def predict_with_confidence(model, tokenizer, sentence: str, aspect: str):
    """Returns (predicted_label, {label: probability, ...})"""
    encoded = tokenizer(sentence, aspect, truncation=True, max_length=CFG.MAX_SEQ_LENGTH, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    prob_dict = {CFG.ID2LABEL[i]: float(probs[i]) for i in range(len(CFG.LABELS))}
    predicted_label = max(prob_dict, key=prob_dict.get)
    return predicted_label, prob_dict


def print_prediction(sentence: str, aspect: str, predicted: str, probs: dict, true_label: str = None):
    print(f"\n  Sentence : {sentence}")
    print(f"  Aspect   : {aspect}")
    print(f"  Predicted: {predicted.upper()}")
    print(f"  Confidence breakdown:")
    for label in CFG.LABELS:
        bar_len = int(probs[label] * 30)
        bar = "#" * bar_len
        marker = " <-- predicted" if label == predicted else ""
        print(f"    {label:10} {probs[label]*100:5.1f}% {bar}{marker}")
    if true_label:
        correct = "CORRECT" if predicted == true_label else "INCORRECT"
        print(f"  (True label: {true_label} -- {correct})")


def main():
    set_global_seed()

    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("BUILT-IN EXAMPLES (one per domain)")
    print("=" * 60)
    for example in BUILT_IN_EXAMPLES:
        predicted, probs = predict_with_confidence(model, tokenizer, example["sentence"], example["aspect"])
        print(f"\n[{example['domain']}]")
        print_prediction(example["sentence"], example["aspect"], predicted, probs)

    print("\n\n" + "=" * 60)
    print("INTERACTIVE MODE -- type your own sentence + aspect")
    print("(type 'quit' at any prompt to stop)")
    print("=" * 60)

    while True:
        sentence = input("\nEnter an Urdu sentence: ").strip()
        if sentence.lower() == "quit":
            break
        aspect = input("Enter the aspect (a word/phrase from that sentence): ").strip()
        if aspect.lower() == "quit":
            break

        predicted, probs = predict_with_confidence(model, tokenizer, sentence, aspect)
        print_prediction(sentence, aspect, predicted, probs)

    print("\nDemo ended.")


if __name__ == "__main__":
    main()
