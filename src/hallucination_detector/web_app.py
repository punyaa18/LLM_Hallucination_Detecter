"""Minimal local web app for hallucination visualization."""

from pathlib import Path
from typing import Optional
from threading import Lock

from flask import Flask, jsonify, render_template, request

from config.config import DetectorConfig
from .detector import HallucinationDetector
from .visual_signals import report_to_visual_payload


def create_app(config: Optional[DetectorConfig] = None) -> Flask:
    """Create and configure Flask application."""
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    detector = None
    detector_lock = Lock()

    def get_detector() -> HallucinationDetector:
        nonlocal detector
        if detector is None:
            with detector_lock:
                if detector is None:
                    detector = HallucinationDetector(config or DetectorConfig(verbose=False))
        return detector

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/analyze")
    def analyze():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"error": "Text is required"}), 400

        try:
            report = get_detector().detect(text)
            return jsonify(report_to_visual_payload(report))
        except Exception as error:
            return jsonify({"error": str(error)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8080, debug=False)
