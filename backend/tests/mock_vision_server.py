import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class VisionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))

        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "brand_name": "Acme Spirits",
                                "class_type": "Vodka",
                                "alcohol_content": "40% vol",
                                "net_contents": "750 ml",
                                "warning_statement": "Contains sulfites.",
                            }
                        )
                    }
                }
            ]
        }

        body = json.dumps(response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8001), VisionHandler)
    print("Mock vision server listening on http://127.0.0.1:8001")
    server.serve_forever()
