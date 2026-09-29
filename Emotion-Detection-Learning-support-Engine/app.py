from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

tips = {
    "Happy": "Keep learning and maintain your positive attitude!",
    "Sad": "Take a short break and talk to someone you trust.",
    "Angry": "Take a deep breath and relax before studying.",
    "Stressed": "Study one topic at a time and take short breaks.",
    "Relaxed": "Great! This is a good time to learn something new.",
    "Confused": "Revise the topic slowly and ask for help if needed.",
    "Motivated": "Excellent! Keep working towards your goals."
}

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = ""
    tip = ""

    if request.method == "POST":
        text = request.form["text"]
        text_vector = vectorizer.transform([text])
        emotion = model.predict(text_vector)[0]
        tip = tips.get(emotion, "")

    return render_template("index.html", emotion=emotion, tip=tip)

if __name__ == "__main__":
    app.run(debug=True)
