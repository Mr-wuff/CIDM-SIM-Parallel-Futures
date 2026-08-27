import json,socket
class UEWorldStateUDPBridge:
    def __init__(self,host="127.0.0.1",port=17777):
        self.addr=(host,port); self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def send(self,payload):
        self.sock.sendto(json.dumps(payload,ensure_ascii=False).encode("utf-8"),self.addr)
