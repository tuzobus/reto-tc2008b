# server_batch.py
# IMPORTANTE
# Ejecutar los siguientes comandos antes de correr el servidor
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# export NO_PROXY=127.0.0.1,localhost
# python3 server_batch.py
#
# Y si flask no está instalado:
# pip install flask flask-cors

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
#from rescue_model import RescueModel
from rescue_model_smart import RescueModel

app = Flask(__name__)
CORS(app)

CONFIG_PATH = Path("config.json")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/config")
def get_config():
    return jsonify(load_config())


@app.get("/run")
def run_full():
    cfg = load_config()

    max_steps = int(request.args.get("max_steps", "1000"))
    seed = request.args.get("seed", None)
    if seed is not None:
        try:
            seed = int(seed)
        except:
            seed = None

    model = RescueModel(str(CONFIG_PATH))

    steps_run = 0
    while model.running and steps_run < max_steps:
        model.step()
        steps_run += 1

    result = "win" if model.hostages_rescued >= 7 else (
        "lose_lost" if model.hostages_lost >= 4 else
        ("lose_collapse" if model.structural_damage >= 25 else "timeout")
    )

    simlog = model.logger.to_simlog(
        result=result,
        rescued=model.hostages_rescued,
        lost=model.hostages_lost,
        damage=model.structural_damage,
        meta={"rows": cfg["rows"], "cols": cfg["cols"]}
    )
    return jsonify(simlog)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
