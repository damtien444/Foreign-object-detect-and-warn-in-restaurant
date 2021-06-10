import  socket

def flag():
    while True:
        try:
            s=socket.socket();
            #host = '169.254.34.90' #dia chi con rass
            host = '192.168.1.9'  # dia chi con rass
            #host = '169.254.204.202'
            port=4200
            s.connect((host,port))
            flag = int((s.recv(2048)).decode())
            if flag == 1:
                break
            s.close()
        except Exception as e:
            print(e)

