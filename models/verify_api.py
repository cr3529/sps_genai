"""Exercise a real running server, including the original assignment endpoints.

Run: uv run python verify_api.py --url http://127.0.0.1:8014
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from http.client import RemoteDisconnected
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8014")
    args = parser.parse_args()
    records = []
    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    def request(path, expected=200, body=None, content_type=None):
        headers = {"Content-Type": content_type} if content_type else {}
        req = Request(args.url + path, data=body, headers=headers)
        started = time.time()
        try:
            response = urlopen(req, timeout=180)
        except HTTPError as error:
            response = error
        data = response.read()
        record = dict(path=path, status=response.code, expected=expected,
                      seconds=round(time.time()-started, 3), bytes=len(data))
        records.append(record)
        assert response.code == expected, (record, data[:500])
        print(record, flush=True)
        return data, response.headers

    for attempt in range(60):
        try:
            urlopen(args.url + "/", timeout=2).close()
            break
        except (URLError, TimeoutError, RemoteDisconnected, ConnectionResetError):
            time.sleep(1)
    else:
        raise RuntimeError("Server did not become ready")
    request("/")
    info = json.loads(request("/generators/info")[0])
    assert set(info) == {"energy", "diffusion"}
    for kind, steps in (("diffusion", 100), ("energy", 256)):
        path = f"/{kind}/generate?num_samples=4&steps={steps}&seed=7"
        png, headers = request(path)
        assert headers.get_content_type() == "image/png"
        image = Image.open(io.BytesIO(png))
        assert image.format == "PNG" and image.mode == "RGB"
        (docs / f"assignment4_{kind}_api.png").write_bytes(png)
        values = json.loads(request(f"/{kind}/generate/pixels?num_samples=1&steps={steps}&seed=7")[0])
        assert values["shape"] == [3, 32, 32]
        images = values["images"]
        assert len(images) == 1 and len(images[0]) == 3
        assert all(len(channel)==32 and all(len(row)==32 for row in channel) for channel in images[0])
        flat = [value for channel in images[0] for row in channel for value in row]
        assert all(0 <= value <= 1 for value in flat)
        repeated, _ = request(path)
        assert hashlib.sha256(repeated).digest() == hashlib.sha256(png).digest()
        different, _ = request(path.replace("seed=7", "seed=8"))
        assert different != png
        request(f"/{kind}/generate?num_samples=0", 422)
        request(f"/{kind}/generate?num_samples=17", 422)
        request(f"/{kind}/generate?steps=9999", 422)
        request(f"/{kind}/generate?seed=-1", 422)
    spec = json.loads(request("/openapi.json")[0])
    assert "/energy/generate" in spec["paths"] and "/diffusion/generate" in spec["paths"]
    request("/docs")
    request("/generate", body=json.dumps({"start_word":"bigram", "length":5}).encode(), content_type="application/json")
    embedding = json.loads(request("/embedding", body=b'{"word":"apple"}', content_type="application/json")[0])
    assert embedding["dimension"] == 300
    classes = json.loads(request("/classify/classes")[0])
    assert len(classes["classes"]) == 10
    request("/gan/generate?num_samples=4&seed=7")
    values = json.loads(request("/gan/generate/pixels?num_samples=1&seed=7")[0])
    assert values["shape"] == [1, 28, 28]
    # A generated CIFAR image is a valid test upload; classification correctness is
    # not asserted because it is not a labeled held-out photograph.
    png = (docs / "assignment4_diffusion_api.png").read_bytes()
    boundary = "Assignment4Boundary"
    prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.png"\r\n'
              'Content-Type: image/png\r\n\r\n').encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    classified = json.loads(request("/classify", body=prefix+png+suffix,
                                    content_type=f"multipart/form-data; boundary={boundary}")[0])
    assert classified["class"] in classes["classes"]
    request("/classify", expected=400, body=prefix+b"not an image"+suffix,
            content_type=f"multipart/form-data; boundary={boundary}")
    result = dict(server=args.url, checks=len(records), passed=True,
                  checkpoints=info, results=records)
    (docs / "assignment4_api_verification.json").write_text(json.dumps(result, indent=2))
    print(f"PASS: {len(records)} live HTTP checks", flush=True)


if __name__ == "__main__":
    main()
