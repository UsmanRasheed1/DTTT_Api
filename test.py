from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello, world!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
