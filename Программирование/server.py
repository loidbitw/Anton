import asyncore
import asynchat
import socket
import multiprocessing
import logging
import mimetypes
import os
import urllib
import argparse
import time
import re
from urllib import parse


def url_normalize(path): #проверка URL и исправление
    if path.startswith("."):
        path = "/" + path
    while "../" in path:
        p1 = path.find("/..")
        p2 = path.rfind("/", 0, p1)
        if p2 != -1:
            path = path[:p2] + path[p1+3:]
        else:
            path = path.replace("/..", "", 1)
    path = path.replace("/./", "/")
    path = path.replace("/.", "")
    return path

class AsyncHTTPServer(asyncore.dispatcher): #сервер принимает подключение

    def __init__(self, host="127.0.0.1", port=9000): 
        super().__init__()
        self.create_socket()
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accepted(self, sock, addr): #принял соединение
        print("Incoming connection from"+str(addr))
        AsyncHTTPRequestHandler(sock)

    def serve_forever(self): #запускаем на пост время
        asyncore.loop()

def get_bytes(s):
    return bytes(str(s), 'utf-8')

class AsyncHTTPRequestHandler(asynchat.async_chat): #класс обработчика запросов

    def __init__(self, sock): #инициализация
        super().__init__(sock)
        self.set_terminator(b"\r\n\r\n")
        self.term = "\r\n"
        self.headers = {}
        self.response_headers = {
            'Host' : '127.0.0.1',
            'Server' : 'ServerName',
            'Date' : self.date_time_string(),
        }

        self.protocol_version = '1.1'
        self.reading_headers = False

        self.text_extensions = ('txt', 'html', 'css', 'js')
        self.images_extensions = ('jpg', 'jpeg', 'png', 'gif')

        self.responses = {
            200: ('OK', 'Request fulfilled, document follows'),
            400: ('Bad Request',
                'Bad request syntax or unsupported method'),
            403: ('Forbidden',
                'Request forbidden -- authorization will not help'),
            404: ('Not Found', 'Nothing matches the given URI'),
            405: ('Method Not Allowed',
                'Specified method is invalid for this resource.'),
        }

    def collect_incoming_data(self, data): 
        print("Incoming data: "+str(data))
        self._collect_incoming_data(data)

    def found_terminator(self): #если нашли конец, то обрабатываем данные
        self.parse_request()

    def parse_request(self):         #парсим запрос
        if not self.reading_headers:
            if not self.parse_headers():
                self.send_error(400, self.responses[400])       
                self.handle_close()
            if self.method == 'POST':
                content_len = self.headers.get("Content-Length", 0)
                if content_len > 0:
                    self.set_terminator(int(content_len)) 
                else: 
                 
                    self.handle_request() 
            else:           
                self.set_terminator(None)     
                self.handle_request()
        else:
            self.set_terminator(None)
            self.request_body = self._get_data()
            self.handle_request()


    def parse_headers(self): #парсим заголовки
        def make_dict(headers):
            result = {}
            for header in headers:
                key = header.split(':')[0].lower()
                value = header[len(key) + 2:] 
                result[key] = value
            return result

        raw = self._get_data().decode()

        # метод парсинга, узнаем get или post, используя регул выражение и находим метод. если нет метода обработки, то возвращем 0
        self.method = re.findall('^[A-Z]+', raw)[0]
        print('Method: '+self.method)
        if not hasattr(self, 'do_' + self.method):
            self.send_error(405)
            return False

        # парсинг протокола
        matches = re.findall('\/(1.1|1.0|0.9)\\r\\n', raw)
        if len(matches) == 0:
            return False
        self.protocol_version = matches[0].strip().replace('/', '')
        print("protocol: "+self.protocol_version)

        # parsing request uri
        expression = '^'+self.method+'(.*?)HTTP/'+self.protocol_version
        matches = re.findall(expression, raw)
        if len(matches) == 0:
            return False
        uri = matches[0]
        self.uri = uri[1:-1] # removing spaces from both sides
        self.uri = parse.unquote(self.uri) # URLDecode
        self.uri = self.edit_path(self.uri)
        print("uri: '"+self.uri+"'")

        # парсинг заголовков

        if self.method in ('GET', 'HEAD'):
            # 'GET / HTTP/1.1\r\nHost: 127.0.0.1:9000\r\nUser-Agent: curl/7.49.1\r\nAccept: */*'
            self.headers = make_dict(raw.split(self.term)[1:])

            if self.protocol_version == '1.1' and 'host' not in self.headers:
                return False

            # extracting query string
            if '?' in self.uri:
                temp = self.uri
                self.uri = re.findall('^(.*?)\?', temp)[0]
                self.query_string = temp[len(self.uri) + 1:] # 'http://mail.ru/get?a=b' <- len(uri) + 1 (?)
                print("uri: '"+self.uri+"', query_string: '"+self.query_string+"'")

        elif self.method == 'POST':
            # 'GET / HTTP/1.1\r\nHost: 127.0.0.1:9000\r\nUser-Agent: curl/7.49.1\r\nAccept: */*\r\n\r\nBodddyyyy\r\n\r\n'
            head = raw.split(self.term * 2)[:1][0]
            self.headers = make_dict(head.split(self.term)[1:])

            if 'content-length' not in self.headers:
                return False

        return True


    def handle_request(self): #запускаем метод, который нужен
        method_name = 'do_' + self.method
        if not hasattr(self, method_name):
            self.send_error(405)
            self.handle_close()
            return
        handler = getattr(self, method_name)
        handler()


    def send_error(self, code, message=None): #если ошибка
        try:
            short_msg, long_msg = self.responses[code]
        except KeyError:
            short_msg, long_msg = 'error', 'Error'
        if message is None:
            message = short_msg

        self.send_response(code, message)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Connection", "close")
        self.end_headers()


    def end_headers(self):
        self.push(bytes(str("\r\n"), 'utf-8'))


    def send_response(self, code, content=''): #отправляем ответ
        
        print("respond with code: "+str(code))

        try:
            message, _ = self.responses[code]
        except KeyError:
            message = 'mistake'

        self.push(get_bytes("HTTP/"+self.protocol_version+" "+str(code)+" "+message))
        self.add_terminator()
        # may be not
        print(self.response_headers)
        for key, value in self.response_headers.items():
            self.send_header(key.title(), value)
        self.add_terminator()

        if self.method == "POST":
            self.push(get_bytes(self.body))
        else:
            if len(content) > 0:
                self.send(content)

        self.add_terminator()
        self.add_terminator()
        self.handle_close()


    def date_time_string(self): #передаем в свое ответе время и дату ответа
        weekdayname = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
        monthname = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
        year, month, day, hh, mm, ss, wd, y, z = time.gmtime(time.time())
        return str(weekdayname[wd]) + ", " + str(day) + " " + str(monthname[month]) + " " + str(year) + " " + str(
            hh) + ":" + str(mm) + ":" + str(ss) + " GMT"


    def send_header(self, keyword, value):
        self.push(get_bytes(str(keyword)+": "+str(value)))
        self.add_terminator()        


    def add_terminator(self):
        self.push(get_bytes(self.term))



    def do_GET(self, without_content=False): 
        # find document by uri
        print("do_GET: uri == '"+self.uri+"'")


        valid_extensions = self.text_extensions + self.images_extensions
        max_extension_length = len(max(valid_extensions, key=len)) + 1 # 1 is for dot

        is_file = '.' in self.uri[-max_extension_length:]

        if not is_file: # wtf??
            self.send_error(403)
            return

        if os.path.exists(self.uri):
            extension = self.uri.split(".")[-1:][0]
            if extension in valid_extensions:
                self.response_headers['content-type'] = self.make_content_type_header(extension)
                reading_mode = 'r' if extension in self.text_extensions else 'rb'
                with open(self.uri, reading_mode) as f:
                    data = f.read()
                    self.response_headers['content-length'] = len(data)
                if without_content: # called from do_HEAD
                    data = ''
                self.send_error(200, data)
            else:
                self.send_error(415)
        else:
            self.send_error(404)

    def convert_extension_to_content_type_ending(self, s): 
        replacements = [('txt', 'plain'), ('js', 'javascript'), ('jpg', 'jpeg')]
        for item in replacements:
            s = s.replace(item[0], item[1])
        return s

    def make_content_type_header(self, extension):
        first_part = 'text' if extension in self.text_extensions else 'image'
        extension = self.convert_extension_to_content_type_ending(extension)
        return str(first_part)+"/"+str(extension)


    def handle_data(self):
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)

    def do_POST(self):
        if self.uri.endswith('.html'):
            # naive
            # logging.debug("do_POST: Sending error 400 because of self.uri.endswith('.html')")
            self.send_error(400)
        else:
            self.response_headers['content-length'] = len(self.body)
            self.send_error(200)

    def do_HEAD(self):
        self.do_GET(without_content=True)


    def edit_path(self, path):
        if path.startswith("."):
            path = "/" + path
        while "../" in path:
            p1 = path.find("/..")
            p2 = path.find("/", 0, p1)
            if p2 != -1:
                path = path[:p2] + path[p1 + 3:]
            else:
                path = path.replace("/..", "", 1)
        path = path.replace("/./", "/")
        path = path.replace("/.", "")

        if path == '/':
            path += 'index.html'
        if path.startswith('/'): # removing slash from beginning
            path = path[1:]
        return path


def parse_args():
    parser = argparse.ArgumentParser("Simple asynchronous web-server")
    parser.add_argument("--host", dest="host", default="127.0.0.1")
    parser.add_argument("--port", dest="port", type=int, default=9000)
    parser.add_argument("--log", dest="loglevel", default="info")
    parser.add_argument("--logfile", dest="logfile", default=None)
    parser.add_argument("-w", dest="nworkers", type=int, default=1)
    parser.add_argument("-r", dest="document_root", default=".")
    return parser.parse_args()

def run():
    server = AsyncHTTPServer(host="127.0.0.1", port=9000)
    server.serve_forever()

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        filename=args.logfile,
        level=getattr(logging, args.loglevel.upper()),
        format="%(name)s: %(process)d %(message)s")
    log = logging.getLogger(__name__)

    DOCUMENT_ROOT = args.document_root
    for _ in range(args.nworkers):
        p = multiprocessing.Process(target=run)
        p.start()

