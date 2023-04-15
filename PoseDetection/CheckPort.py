import serial.tools.list_ports
import serial

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portlist = []

print("Open Ports:")

for onePort in ports:
    portlist.append(str(onePort))
    print(str(onePort))