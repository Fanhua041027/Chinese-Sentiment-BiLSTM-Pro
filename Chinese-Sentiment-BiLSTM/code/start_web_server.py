"""
启动 Web 服务器
"""
import http.server
import socketserver

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
    print(f'Serving at http://127.0.0.1:{PORT}')
    httpd.serve_forever()