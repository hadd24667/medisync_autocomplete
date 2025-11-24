from flask import Flask, request, jsonify
from flask_cors import CORS
from Medisync_Autocomplete import autocomplete_icd, autocomplete_atc, icd_engine, atc_engine

app = Flask(__name__)
CORS(app)

@app.get("/autocomplete")
def api_autocomplete():
    q = request.args.get("q", "").strip()
    mode = request.args.get("type", "icd")  # icd / atc

    if not q:
        return jsonify({"results": [], "source": "empty", "latency_ms": 0})

    if mode == "atc":
        results, source, ms = autocomplete_atc(q)
    else:
        results, source, ms = autocomplete_icd(q)

    return jsonify({
        "results": results,
        "source": source,
        "latency_ms": ms
    })


@app.post("/clear_cache")
def clear_cache():
    icd_cleared = icd_engine.clear_cache()
    atc_cleared = atc_engine.clear_cache()
    return jsonify({
        "cleared_icd": icd_cleared,
        "cleared_atc": atc_cleared
    })


if __name__ == "__main__":
    print("🚀 SmartEMR Autocomplete API running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
