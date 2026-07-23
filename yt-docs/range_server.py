#!/usr/bin/env python3
"""Simple HTTP server with correct Range request support."""
import http.server, socketserver, sys, os

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path) or os.path.isdir(path):
            return super().send_head()
        fsize = os.path.getsize(path)
        range_header = self.headers.get('Range')
        if range_header and range_header.startswith('bytes='):
            spec = range_header[6:]
            try:
                start_str, end_str = spec.split('-')
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else fsize - 1
                end = min(end, fsize - 1)
                if start > end or start >= fsize:
                    self.send_error(416, "Range Not Satisfiable")
                    return None
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{fsize}')
                self.send_header('Content-Length', str(end - start + 1))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Type', 'application/octet-stream')
                self.end_headers()
                f = open(path, 'rb')
                f.seek(start)
                return f  # note: file-like, will be read by send_body
            except (ValueError, IOError):
                self.send_error(400, "Bad Range")
                return None
        # No range: full file
        self.send_response(200)
        self.send_header('Content-Length', str(fsize))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        return open(path, 'rb')

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18432
    directory = sys.argv[2] if len(sys.argv) > 2 else '.'
    os.chdir(directory)
    with socketserver.ThreadingTCPServer(("", port), RangeHandler) as httpd:
        print(f"Serving {directory} on port {port}")
        httpd.serve_forever()
