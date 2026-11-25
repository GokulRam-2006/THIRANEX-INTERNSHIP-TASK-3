from flask import Flask, request, jsonify, render_template_string
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -----------------------------
# TRAIN MACHINE LEARNING MODEL
# -----------------------------

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

names = iris.target_names


# -----------------------------
# HTML + CSS + JAVASCRIPT
# -----------------------------

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>CyberLens AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;

    background:
    radial-gradient(circle at top left,#182b55,#070b16 45%),
    #070b16;

    color: white;

    min-height: 100vh;

    display: flex;
    justify-content: center;
    align-items: center;

    padding: 25px;
}

.container {

    width: 700px;
    max-width: 100%;

    background: rgba(15,23,42,.94);

    border: 1px solid #263653;

    border-radius: 22px;

    padding: 30px;

    box-shadow:
    0 20px 60px #0009;

}

.header {

    text-align: center;

    margin-bottom: 25px;
}

.logo {

    font-size: 45px;
}

h1 {

    margin: 5px;

    font-size: 32px;

    background:
    linear-gradient(
        90deg,
        #00d9ff,
        #8b5cf6,
        #00ffa3
    );

    -webkit-background-clip: text;

    color: transparent;
}

.subtitle {

    color: #8d9bb0;

    font-size: 14px;
}


.status {

    display: inline-block;

    margin-top: 12px;

    padding: 7px 13px;

    border-radius: 20px;

    background: #063d30;

    color: #4cffc3;

    font-size: 12px;
}


.inputs {

    display: grid;

    grid-template-columns:
    1fr 1fr;

    gap: 15px;
}

label {

    color: #9caac0;

    font-size: 13px;

    display: block;

    margin-bottom: 6px;
}

input {

    width: 100%;

    padding: 13px;

    border-radius: 9px;

    border: 1px solid #33445f;

    background: #0a1220;

    color: white;

    font-size: 16px;

    outline: none;
}

input:focus {

    border-color: #00cfff;

    box-shadow:
    0 0 0 3px #00cfff18;
}


button {

    width: 100%;

    margin-top: 22px;

    padding: 14px;

    border: none;

    border-radius: 10px;

    background:
    linear-gradient(
        90deg,
        #007cf0,
        #00c6ff,
        #7657ff
    );

    color: white;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;

    transition: .2s;
}

button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 10px 30px #008cff44;
}


.result {

    margin-top: 25px;

    padding: 25px;

    text-align: center;

    border-radius: 16px;

    background:
    linear-gradient(
        135deg,
        #062b2a,
        #101c31
    );

    border: 1px solid #174d4a;

    display: none;
}

.result-icon {

    font-size: 42px;
}

.result-title {

    color: #7c8ba3;

    font-size: 12px;

    letter-spacing: 2px;

    margin-top: 10px;
}

.result-name {

    font-size: 30px;

    font-weight: bold;

    color: #5fffd0;

    margin: 8px;
}

.confidence {

    font-size: 34px;

    font-weight: bold;

    color: #00d9ff;
}

.bar {

    height: 10px;

    background: #202d42;

    border-radius: 10px;

    margin-top: 12px;

    overflow: hidden;
}

.progress {

    height: 100%;

    width: 0%;

    background:
    linear-gradient(
        90deg,
        #00ffa3,
        #00cfff
    );

    transition: .7s;
}


.probabilities {

    text-align: left;

    margin-top: 20px;
}

.probability {

    margin: 10px 0;
}

.probability-top {

    display: flex;

    justify-content: space-between;

    color: #aab6c8;

    font-size: 13px;

    margin-bottom: 5px;
}

.small-bar {

    height: 6px;

    background: #202d42;

    border-radius: 10px;

    overflow: hidden;
}

.small-progress {

    height: 100%;

    background:
    linear-gradient(
        90deg,
        #7657ff,
        #00cfff
    );
}


.stats {

    display: grid;

    grid-template-columns:
    1fr 1fr;

    gap: 12px;

    margin-top: 20px;
}

.stat {

    padding: 15px;

    border-radius: 12px;

    background: #0b1424;

    border: 1px solid #1d2b41;
}

.stat span {

    display: block;

    color: #718096;

    font-size: 11px;
}

.stat strong {

    display: block;

    margin-top: 5px;

    font-size: 21px;
}


.footer {

    text-align: center;

    color: #56657b;

    font-size: 11px;

    margin-top: 20px;
}


@media(max-width:600px) {

    .inputs {

        grid-template-columns: 1fr;
    }

    h1 {

        font-size: 27px;
    }

}

</style>

</head>


<body>


<div class="container">


<div class="header">

<div class="logo">🛡️</div>

<h1>CyberLens AI</h1>

<div class="subtitle">

Machine Learning Classification System

</div>

<div class="status">

● AI MODEL ONLINE

</div>

</div>


<h3>🔬 Enter Flower Measurements</h3>


<div class="inputs">


<div>

<label>Sepal Length (cm)</label>

<input
type="number"
id="sepal_length"
value="5.8"
step="0.1"
min="4"
max="8">

</div>


<div>

<label>Sepal Width (cm)</label>

<input
type="number"
id="sepal_width"
value="3.0"
step="0.1"
min="2"
max="4.5">

</div>


<div>

<label>Petal Length (cm)</label>

<input
type="number"
id="petal_length"
value="4.0"
step="0.1"
min="1"
max="7">

</div>


<div>

<label>Petal Width (cm)</label>

<input
type="number"
id="petal_width"
value="1.2"
step="0.1"
min="0.1"
max="2.5">

</div>


</div>


<button onclick="predict()">

🚀 ANALYZE WITH AI

</button>


<div id="result" class="result">

<div class="result-icon">

🌸

</div>

<div class="result-title">

AI PREDICTION

</div>

<div id="prediction" class="result-name">

---

</div>

<div id="confidence" class="confidence">

0%

</div>

<div>

AI Confidence

</div>

<div class="bar">

<div id="progress" class="progress"></div>

</div>


<div class="probabilities">

<h4>🧠 Prediction Probability</h4>

<div id="probabilityBox"></div>

</div>

</div>


<div class="stats">

<div class="stat">

<span>MODEL</span>

<strong>Random Forest</strong>

</div>

<div class="stat">

<span>DATASET</span>

<strong>Iris</strong>

</div>

<div class="stat">

<span>ACCURACY</span>

<strong>{{ accuracy }}%</strong>

</div>

<div class="stat">

<span>FEATURES</span>

<strong>4</strong>

</div>

</div>


<div class="footer">

Powered by Python • Flask • Scikit-learn

</div>


</div>


<script>

async function predict() {

    let data = {

        sepal_length:
        document.getElementById(
            "sepal_length"
        ).value,

        sepal_width:
        document.getElementById(
            "sepal_width"
        ).value,

        petal_length:
        document.getElementById(
            "petal_length"
        ).value,

        petal_width:
        document.getElementById(
            "petal_width"
        ).value
    };


    let button =
    document.querySelector("button");

    button.innerText =
    "⏳ ANALYZING...";


    let response =
    await fetch(
        "/predict",
        {

            method: "POST",

            headers: {
                "Content-Type":
                "application/json"
            },

            body:
            JSON.stringify(data)

        }
    );


    let result =
    await response.json();


    document.getElementById(
        "result"
    ).style.display = "block";


    document.getElementById(
        "prediction"
    ).innerText =
    result.prediction.toUpperCase();


    document.getElementById(
        "confidence"
    ).innerText =
    result.confidence + "%";


    document.getElementById(
        "progress"
    ).style.width =
    result.confidence + "%";


    let box =
    document.getElementById(
        "probabilityBox"
    );

    box.innerHTML = "";


    result.probabilities.forEach(
        item => {

            box.innerHTML += `

            <div class="probability">

                <div class="probability-top">

                    <span>
                    ${item.name}
                    </span>

                    <b>
                    ${item.value}%
                    </b>

                </div>

                <div class="small-bar">

                    <div
                    class="small-progress"
                    style="
                    width:${item.value}%;
                    ">
                    </div>

                </div>

            </div>

            `;

        }
    );


    button.innerText =
    "🚀 ANALYZE WITH AI";

}

</script>


</body>

</html>
"""


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():

    return render_template_string(
        HTML,
        accuracy=round(accuracy * 100, 2)
    )


# -----------------------------
# PREDICTION API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    values = [[
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"])
    ]]

    prediction = model.predict(values)[0]

    probabilities = model.predict_proba(values)[0]

    return jsonify({

        "prediction":
        names[prediction],

        "confidence":
        round(
            probabilities[prediction] * 100,
            2
        ),

        "probabilities": [

            {
                "name": names[i],
                "value": round(
                    probabilities[i] * 100,
                    2
                )
            }

            for i in range(3)

        ]

    })


# -----------------------------
# START SERVER
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )