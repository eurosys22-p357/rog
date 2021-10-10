import sys
import time
import operator
interval=0.5



dev =open("/proc/net/dev","r")
lines=dev.readlines()
count=0
for line in lines[2:]:
    intf=line[:line.index(":")].strip()
    if operator.eq(intf[:2],'wl'):
        count+=1
assert count==1
values={}
transmit=0
while True:
    dev.seek(0)
    lines=dev.readlines()
    for line in lines[2:]:
        intf=line[:line.index(":")].strip()
        if operator.ne(intf[:2],'wl'):
            continue
        values[intf]=[int(value) for value in line[line.index(":")+1:].split()]
        new=values[intf][8]
        print(time.time(),(new-transmit)/interval*8/1024/1024, flush=True)
        transmit=new
        break
    time.sleep(interval)
