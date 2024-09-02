
from flask import Flask, request
from deepface import DeepFace
from flask_cors import CORS

import logging

_logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"

@app.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    verification = verifyimg(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    _logger.debug(verification)

    return verification

def verifyimg(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    # try:
    print("img1_path", img1_path)
    print("img2_path", img2_path)
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
        anti_spoofing=anti_spoofing,
    )
    return obj
    # except Exception as err:
    #     tb_str = traceback.format_exc()
    #     return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') # use adhoc for testing

    app.run(debug=True, host="0.0.0.0", port=3000) 