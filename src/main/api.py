"""
API implementation for the ConvFinQA system (WSGI compatible).
"""

from flask import Flask, jsonify, request

from src.main.predict import predict

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "ConvFinQA API is running"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict answer for a financial question.

    Expected JSON body:
    {
        "question": "the question to answer"
    }
    """
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Question is required"}), 400

        question = data["question"]

        print(f"Using model to predict answer for question: {question}")
        answer = predict(question=question)

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
