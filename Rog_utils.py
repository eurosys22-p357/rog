from numpy.lib.histograms import _ravel_and_check_weights
from torch import tensor
from torch._C import dtype
from torch.distributed.distributed_c10d import recv
import socket
import queue
import threading
import pickle
import multiprocessing as mp
import time
from utils import Bar, AverageMeter, accuracy, mkdir_p, savefig
import torch
from math import cos, pi
import torch.distributed as dist
import numpy as np
import math
MAX_RECV_SIZE=4*1024
OK_ACK=1000
class TCPMessageStream:
    NUM_SIZE=4
    def __init__(self,sock:socket.socket,monitor):
        self.sock=sock
        self.monitor=monitor
        self.send_queue=queue.Queue()
        def send_task():
            while True:
                msg=self.send_queue.get()
                msize = len(msg)
                self.sock.send(msize.to_bytes(self.NUM_SIZE,"big")+msg)
        self.send_thread=threading.Thread(target=send_task)
        self.send_thread.start()

        self.recv_queue=queue.Queue()
        def recv_task():
            buffer=bytearray()
            tail="immediately"
            tail_bytes=tail.encode("utf-8")
            while True:
                while len(buffer) < self.NUM_SIZE:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msize=int.from_bytes(buffer[:self.NUM_SIZE],"big")
                buffer=buffer[self.NUM_SIZE:]
                while len(buffer)<msize:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msg =buffer[:msize]
                buffer=buffer[msize:]
                if msg[-len(tail_bytes):]==tail_bytes:
                    self.monitor.put(msg[:-len(tail_bytes)])
                else:
                    self.recv_queue.put(msg)
        self.recv_thread=threading.Thread(target=recv_task)
        self.recv_thread.start()

    def send(self,msg):
        self.send_queue.put(msg)

    def recv(self):
        return self.recv_queue.get()

class UDPMessageStream:
    def __init__(self,sock:socket.socket,congestion_control_min,congestion_control_max,congestion_control,world_size):
        self.sock=sock
        self.min=congestion_control_min/100
        self.max=congestion_control_max/100
        self.world_size=world_size
        self.importance_threshold=0.0
        self.send_queue=queue.Queue()
        self.address_queue=queue.Queue()
        def send_task():
            while True:
                addr=self.address_queue.get()
                while not self.send_queue.empty():
                    msg,importance=self.send_queue.get()
                    if importance>=self.importance_threshold:
                        self.sock.sendto(msg,addr)
                    #print("i send a message to",addr)
        self.send_thread=threading.Thread(target=send_task)
        self.send_thread.start()

        self.recv_queue=queue.Queue()
        def recv_task():    
            while True:
                msg,addr=self.sock.recvfrom(MAX_RECV_SIZE)
                self.recv_queue.put(msg)
                #print("i get a message")
        self.recv_thread=threading.Thread(target=recv_task)
        self.recv_thread.start()

        self.degree=0
        self.thresholds=[0.0 for _ in range(world_size+1)]
        def update_task():    
            while True:
                self.degree=congestion_control.get()
                print("udp degree is",self.degree)
                self.importance_threshold=self.thresholds[self.degree]
        self.update_thread=threading.Thread(target=update_task)
        self.update_thread.start()

    def put(self,msg,importance):
        self.send_queue.put((msg,importance))
        
    def recvfrom(self):
        return self.recv_queue.get()
    def clean_recv(self):
        while not self.recv_queue.empty():
            self.recv_queue.get()
    def recvfrom_nowait(self):
        try:
            return self.recv_queue.get_nowait()
        except:
            return None
    def sendto(self,address,importance):
        importance.sort()
        self.thresholds=[importance[0] for _ in range(self.world_size+1)]
        if self.world_size>1:
            for i in range(self.world_size):
                index=self.max-(i/(self.world_size-1)*(self.max-self.min))
                index=round(len(importance)*(1-index))
                self.thresholds[i+1]=importance[index]
        else:
            index=round(len(importance)*(1-self.min))
            self.thresholds[1]=importance[index]
        self.importance_threshold=self.thresholds[self.degree]
        print("threshold",self.importance_threshold)
        self.address_queue.put(address)

class layer_unit:
    def __init__(self,p_idx,threshold,tensor,mtu_number):
        self.name=p_idx
        self.threshold=threshold
        self.size=tensor.size()
        self.row=[]

        length=tensor.numel()
        num=math.ceil(length/mtu_number)
        for i in range(num):
            start=i*mtu_number
            if i==num-1:
                end=length
            else:
                end=(i+1)*mtu_number
            self.row.append((int(start),int(end)))

    def send_model(self,tensor,udpstream:UDPMessageStream,row_states,world_size,importance,training_step):
        reshaped=tensor.detach().numpy().tobytes()
        row_states_bytes=row_states.tobytes()
        layer_idx=self.name.to_bytes(4,"big")
        rows_id=np.array(range(len(self.row))).tobytes()
        for i in range(len(self.row)):
            important=training_step-min(row_states[i])
            udpstream.put(reshaped[self.row[i][0]*4:self.row[i][1]*4]+row_states_bytes[i*world_size*8:(i+1)*world_size*8]+rows_id[i*8:(i+1)*8]+layer_idx,important)
            importance.append(important)
    def re_send_model(self,tensor,row_states,world_size,send_idx,result):
        reshaped=tensor.detach().numpy().tobytes()
        row_states_bytes=row_states.tobytes()
        layer_idx=self.name.to_bytes(4,"big")
        rows_id=np.array(range(len(self.row))).tobytes()
        for i in send_idx:
            result.append(reshaped[self.row[i][0]*4:self.row[i][1]*4]+row_states_bytes[i*world_size*8:(i+1)*world_size*8]+rows_id[i*8:(i+1)*8]+layer_idx)
    def start_send(self,udpstream:UDPMessageStream,address,importance):
        data=pickle.dumps("ok")
        for _ in range(OK_ACK):
            udpstream.put(data,float("inf"))
        udpstream.sendto(address,importance)
        print("send ok")
    def send_gradient(self,tensor,udpstream:UDPMessageStream,remain_number,importance,row_states,training_step):
        reshaped=tensor.detach().numpy().tobytes()
        one_dim=tensor.view(-1)
        remain_number_bytes=remain_number.tobytes()
        layer_idx=self.name.to_bytes(4,"big")
        rows_id=np.array(range(len(self.row))).tobytes()
        for i in range(len(self.row)):
            important=torch.sum(torch.abs(one_dim[self.row[i][0]:self.row[i][1]])).item()
            if training_step ==min(row_states[i])+self.threshold+1:
                important=float("inf")
                print("**********",training_step,row_states[i])
            udpstream.put(reshaped[self.row[i][0]*4:self.row[i][1]*4]+remain_number_bytes[i*8:(i+1)*8] +rows_id[i*8:(i+1)*8]+layer_idx,important)
            importance.append(important)
    
        
class Parameter_Server:
    def __init__(self,ps_ip,ps_port,world_size,threshold_min,threshold_max,model,optimizer,communication_library,MTU,congestion_control_min,congestion_control_max):
        self.world_size=world_size
        self.model=model
        self.gathered_weight=mp.Queue(maxsize=1000)
        self.lock=threading.Lock()
        self.training_step=[0 for _ in range(world_size)]
        self.min_step=mp.Queue(maxsize=world_size)
        self.communication_library=communication_library
        self.mtu_number=MTU/4

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ps_ip, ps_port))
        sock.listen(world_size)
        self.connection=0

        self.layer_info=[]
        layers=0
        for _ in self.model.parameters():
            layers+=1
        layer=0
        for p_idx,p in enumerate(self.model.parameters()):
            layer+=1
            threshold=round(threshold_min+(threshold_max-threshold_min)*(layer/layers))
            self.layer_info.append(layer_unit(p_idx,threshold,p,self.mtu_number))
            assert p.dtype==torch.float32
        self.row_states=[]
        self.row_number=0
        for i in range(len(self.layer_info)):
            rows=[]
            self.row_number+=len(self.layer_info[i].row)
            for _ in range(len(self.layer_info[i].row)):
                rows.append([0 for _ in range(world_size)])
            self.row_states.append(np.array(rows))

        proc=[]
        self.isupdated=[]
        self.broadcast=[]
        for _ in range(world_size):
            self.isupdated.append(mp.Queue(maxsize=10))
            self.broadcast.append(mp.Queue(maxsize=10))
        t=threading.Thread(target=self.parameter_server_optimizer,args=(optimizer,))
        proc.append(t)
       
        for i in range(world_size):
            client_sock, client_address = sock.accept()
            immediately_queue=mp.Queue()
            client_stream=TCPMessageStream(client_sock,immediately_queue)
            t=threading.Thread(target=self.each_parameter_server,args=(client_stream,client_address,ps_ip,ps_port,i,immediately_queue,congestion_control_min,congestion_control_max))
            proc.append(t)
        for t in proc:
            t.start()
        print("start parameter server",flush=True)
    
    def each_parameter_server(self,client_stream:TCPMessageStream,client_address,ps_ip,ps_port,rank,immediately_queue,congestion_control_min,congestion_control_max):
        congestion_control=mp.Queue(maxsize=1)
        udp_sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind((ps_ip, ps_port+1+rank))
        udp_stream=UDPMessageStream(udp_sock,congestion_control_min,congestion_control_max,congestion_control,self.world_size)
        client_udp_address=[]
        client_udp_address.append(client_address[0])
        client_udp_address.append(client_address[1]+1)
        client_udp_address=tuple(client_udp_address)
        
        
        def background_task(immediately_queue):
            while True:
                msg=pickle.loads(immediately_queue.get())
        background_thread=threading.Thread(target=background_task,args=(immediately_queue,))
        background_thread.start()
    
        def broadcast_task():
            tail="immediately"
            tail_bytes=tail.encode("utf-8")
            while True:
                msg=self.broadcast[rank].get()
                if msg=="congestion":
                    client_stream.send(pickle.dumps("congestion"+str(self.connection))+tail_bytes)
                if msg=="ask for new":
                    ask_for_new=self.broadcast[rank].get()
                    print("ask for new")
                    client_stream.send(pickle.dumps("ask for new")+tail_bytes)
                    client_stream.send(pickle.dumps(ask_for_new)+tail_bytes)

        broadcast_thread=threading.Thread(target=broadcast_task)
        broadcast_thread.start()

        while True:
            msg=pickle.loads(client_stream.recv())
            print(msg)
            if msg=='get':
                client_stream.send(pickle.dumps((self.model,self.layer_info,(ps_ip, ps_port+1+rank))))
            if msg=='ask':
                self.connection+=1
                congestion_control.put(self.connection)
                for i in range(self.world_size):
                    self.broadcast[i].put("congestion")
                if self.communication_library=="gloo":
                    srcrank=pickle.loads(client_stream.recv())
                    for p in self.model.parameters():
                        dist.send(p,srcrank)
                if self.communication_library=="tcp":
                    client_stream.send(pickle.dumps(self.model))
                if self.communication_library=="rog":
                    importance=[]
                    for i,p in enumerate(self.model.parameters()):
                        self.layer_info[i].send_model(p,udp_stream,self.row_states[i],self.world_size,importance,self.training_step[rank])
                    self.layer_info[i].start_send(udp_stream,client_udp_address,importance)
                self.connection-=1
                congestion_control.put(self.connection)
                for i in range(self.world_size):
                    self.broadcast[i].put("congestion")
            if msg=='send':
                self.connection+=1
                congestion_control.put(self.connection)
                for i in range(self.world_size):
                    self.broadcast[i].put("congestion")
                with self.lock:
                    self.training_step[rank]+=1
                if self.communication_library=="gloo":
                    lr=pickle.loads(client_stream.recv())
                    weight=[]
                    for p in self.model.parameters():
                        recv_tensor=torch.ones_like(p)
                        dist.recv(recv_tensor,srcrank)
                        weight.append(recv_tensor)
                    success="ok"
                if self.communication_library=="tcp":
                    weight,lr=pickle.loads(client_stream.recv())
                    success="ok"
                if self.communication_library=="rog":
                    lr=pickle.loads(client_stream.recv())
                    result=[]
                    udp_stream.clean_recv()
                    far_away_end=True
                    while True:
                        if far_away_end:
                            data=udp_stream.recvfrom()
                            if len(data)==12:
                                print(pickle.loads(data))
                                far_away_end=False
                                continue
                        else:
                            data=udp_stream.recvfrom_nowait()
                            if data==None:
                                break
                            if len(data)==12:
                                far_away_end=False
                                continue
                        result.append(data)
                    weight,success=self.gather_gradient(result)
                self.gathered_weight.put((weight,lr,rank))
                msg=self.isupdated[rank].get()
                assert msg=="ok"
                client_stream.send(pickle.dumps(success))
                self.connection-=1
                congestion_control.put(self.connection)
                for i in range(self.world_size):
                    self.broadcast[i].put("congestion")
            if msg=="finish":
                with self.lock:
                    myself=self.training_step[rank]
                    now=min(self.training_step)
                while now!=0 and myself!=now:
                    now=self.min_step.get()
                self.training_step[rank]=0
                client_stream.send(pickle.dumps("ok"))
            if msg=='need update':
                need_updates=pickle.loads(client_stream.recv())
                stall=[]
                ask_for_new=[]
                unsent=[]
                for need_update in need_updates:
                    i,j,min_step=need_update
                    if min(self.training_step)<min_step:
                        stall.append(need_update)
                    else:
                        if min(self.row_states[i][j])<min_step:
                            ask_for_new.append(need_update)
                        else:
                            unsent.append(need_update)
                print("stall",len(stall))
                print("ask for new",len(ask_for_new))
                print("unsent",len(unsent))
                result=[]
                idx=0
                for i,p in enumerate(self.model.parameters()):
                    send_idx=[]
                    while idx<len(unsent) and unsent[idx][0]==i:
                        send_idx.append(unsent[idx][1])
                        idx+=1
                    self.layer_info[i].re_send_model(p,self.row_states[i],self.world_size,send_idx,result)
                requirement=[[] for _ in range(self.world_size)]
                for each in ask_for_new:
                    threshold=self.layer_info[each[0]].threshold
                    iteration=each[2]
                    for i in range(self.world_size):
                        if self.row_states[each[0]][each[1]][i]+threshold<iteration:
                            requirement[i].append(each)
                for each in stall:
                    threshold=self.layer_info[each[0]].threshold
                    iteration=each[2]
                    for i in range(self.world_size):
                        if self.row_states[each[0]][each[1]][i]+threshold<iteration:
                            requirement[i].append(each)
                for i in range(self.world_size):
                    self.broadcast[i].put("ask for new")
                    self.broadcast[i].put(requirement[i])
                client_stream.send(pickle.dumps())
                result.append(pickle.loads(client_stream.recv()))

                print("send:",len(result))
                client_stream.send(pickle.dumps(result))




    def parameter_server_optimizer(self,optimizer):
        while True:
            weight,lr,rank=self.gathered_weight.get()
            for p,g in zip(self.model.parameters(),weight):
                p.grad=g/self.world_size
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
            self.isupdated[rank].put("ok")
    def gather_gradient(self,result):
        weight=[]
        success=[]
        each_layer=[[] for _ in range(len(self.layer_info))]
        for each_data in result:
            rank=int.from_bytes(each_data[-4:],"big")
            each_layer[rank].append(each_data)
        for i,p in enumerate(self.model.parameters()):
            reshaped=torch.zeros_like(p.view(-1))
            this_layer=self.layer_info[i]
            for each_row in each_layer[i]:
                idx=np.frombuffer(each_row[-12:-4],dtype=int)[0]
                remain_number=np.frombuffer(each_row[-20:-12],dtype=int)
                reshaped.data[this_layer.row[idx][0]:this_layer.row[idx][1]]=reshaped.data[this_layer.row[idx][0]:this_layer.row[idx][1]]+torch.tensor(np.frombuffer(each_row[:-20],dtype=np.float32))
                self.row_states[i][idx]+=remain_number
                success.append((i,idx))
            weight.append(reshaped.view(self.layer_info[i].size))
        print(f"recvice row number: {len(result)} {self.row_number} {len(result)/self.row_number*100}")
        return weight,success   

class Local_Worker:
    def __init__(self,args,model,ps_ip,ps_port,train_loader, train_loader_len,val_loader, val_loader_len, criterion,lr,communication_library):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1) # waiting for ps start
        sock.connect((ps_ip, ps_port))
        self.immediately_queue=mp.Queue()
        self.sock = TCPMessageStream(sock,self.immediately_queue)
        print("conneted to parameter server", flush=True)
        udp_sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind((sock.getsockname()[0], sock.getsockname()[1]+1))
        self.congestion_degree=0
        self.congestion_control=mp.Queue(maxsize=1)
        self.udpsock=UDPMessageStream(udp_sock,args.congestion_control_min,args.congestion_control_max,self.congestion_control,args.world_size-1)
        
        
        self.args=args
        self.model=model
        self.criterion=criterion
        self.train_loader=train_loader
        self.train_loader_len=train_loader_len
        self.val_loader=val_loader
        self.val_loader_len=val_loader_len
        self.lr=lr
        self.communication_library=communication_library

        def background_task():
            while True:
                msg=pickle.loads(self.immediately_queue.get())
                if msg[:10]=="congestion":
                    self.congestion_degree=int(msg[10:])
                    self.congestion_control.put(self.congestion_degree)
                if msg=="stall":
                    self.push_update()
                if msg=="ask for new":
                    self.push_update()
        self.background_thread=threading.Thread(target=background_task)
        self.background_thread.start()
        
    def update_model(self,data):
        each_layer=[[] for _ in range(len(self.layer_info))]
        for each_data in data:
            rank=int.from_bytes(each_data[-4:],"big")
            each_layer[rank].append(each_data)
        for i,p in enumerate(self.model.parameters()):
            reshaped=p.view(-1)
            this_layer=self.layer_info[i]
            for each_row in each_layer[i]:
                idx=np.frombuffer(each_row[-12:-4],dtype=int)[0]
                states=np.frombuffer(each_row[-12-(self.args.world_size-1)*8:-12],dtype=int)
                self.row_states[i][idx]=states
                reshaped.data[this_layer.row[idx][0]:this_layer.row[idx][1]]=torch.tensor(np.frombuffer(each_row[:-12-(self.args.world_size-1)*8],dtype=np.float32))
        print(f"recvice row number: {len(data)} {self.row_number} {len(data)/self.row_number*100}")
    def check_threshold(self,iteration):
        print("check for threshold") 
        waiting_for_new=[]
        for i in range(len(self.row_states)):
            threshold=self.layer_info[i].threshold
            for j in range(len(self.row_states[i])):
                min_step=min(self.row_states[i][j])
                if iteration>min_step+threshold:
                    waiting_for_new.append((i,j,iteration))
        if waiting_for_new!=[]:
            self.sock.send(pickle.dumps("need update"))
            self.sock.send(pickle.dumps(waiting_for_new))
            result=pickle.loads(self.sock.recv())
            self.update_model(result)
            self.check_threshold(iteration)
        print("pass check threshold")
               
    def pull_model(self,iteration):
        print("start ask for new model",flush=True)
        
        self.sock.send(pickle.dumps("ask"))
        if self.communication_library=="gloo":
            start=time.time()
            self.sock.send(pickle.dumps(dist.get_rank()))
            for p in self.model.parameters():
                dist.recv(p,0)
            end=time.time()
        if self.communication_library=="tcp":
            start=time.time()
            self.model=pickle.loads(self.sock.recv())
            end=time.time()
        if self.communication_library=="rog":
            far_away_end=True
            self.udpsock.clean_recv()
            result=[]
            start=time.time()
            while True:
                if far_away_end:
                    data=self.udpsock.recvfrom()
                    if len(data)==12:
                        print(pickle.loads(data))
                        far_away_end=False
                        continue
                else:
                    data=self.udpsock.recvfrom_nowait()
                    if data==None:
                        break
                    if len(data)==12:
                        far_away_end=False
                        continue
                result.append(data)
            end=time.time()
            self.update_model(result)
        self.check_threshold(iteration)
        self.model.train()
        print("complete",end-start,flush=True)
        return end-start
    def push_update(self,iteration):
        print("start push update",flush=True)
        if self.communication_library=="gloo" or self.communication_library=="tcp":
            weight=[]
            for p in self.model.parameters():
                weight.append(p.grad.detach())
        start=time.time()
        self.sock.send(pickle.dumps("send"))
        if self.communication_library=="gloo":
            self.sock.send(pickle.dumps(self.lr))
            for p in self.model.parameters():
                dist.send(p.grad.detach(), 0)
        if self.communication_library == "tcp":
            self.sock.send(pickle.dumps((weight, self.lr)))
        if self.communication_library=="rog":
            self.sock.send(pickle.dumps(self.lr))
            importance=[]
            for i,p in enumerate(self.model.parameters()):
                self.remain[i].data=self.remain[i].data+p.grad
                self.remain_number[i]+=1
                self.layer_info[i].send_gradient(self.remain[i],self.udpsock,self.remain_number[i],importance,self.row_states[i],iteration)
            self.layer_info[i].start_send(self.udpsock,self.udpaddress,importance)
        results = pickle.loads(self.sock.recv())
        for success in results:
            layer_idx=success[0]
            row_idx=success[1]
            p=self.remain[layer_idx]
            p.data[self.layer_info[layer_idx].row[row_idx][0]:self.layer_info[layer_idx].row[row_idx][1]]=torch.zeros_like(p[self.layer_info[layer_idx].row[row_idx][0]:self.layer_info[layer_idx].row[row_idx][1]])
            self.remain_number[layer_idx][row_idx]=0



        end = time.time()
        print("complete",flush=True)
        return end - start
        
    def get_model(self):
        self.sock.send(pickle.dumps("get"))
        self.model,self.layer_info,self.udpaddress=pickle.loads(self.sock.recv())
        self.model.train()
        self.row_states=[]
        self.row_number=0
        for i in range(len(self.layer_info)):
            rows=[]
            self.row_number+=len(self.layer_info[i].row)
            for _ in range(len(self.layer_info[i].row)):
                rows.append([0 for _ in range(self.args.world_size-1)])
            self.row_states.append(np.array(rows))
        self.remain=[]
        self.remain_number=[]
        for i,p in enumerate(self.model.parameters()):
            self.remain.append(torch.zeros_like(p))
            self.remain_number.append(np.array([0 for _ in range(len(self.layer_info[i].row))]))

        print("get new model")
    def train(self, epoch, start_time):
        bar = Bar('Processing', max=self.train_loader_len)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        

        end = time.time()

        self.get_model()
        for i, (input, target) in enumerate(self.train_loader):
            self.adjust_learning_rate(epoch, i, self.train_loader_len)

            # measure data loading time
            data_time.update(time.time() - end)

            # target = target.cuda(non_blocking=True)

            # compute output
            pull_model_time = self.pull_model(i)
            start = time.time()
            output = self.model(input)
            loss = self.criterion(output, target)

            # compute gradient and do SGD step
            
            loss.backward()
            end1 = time.time()
            if (self.args.fix-end1+start>0):
                time.sleep(self.args.fix-end1+start)
            end2=time.time()
            push_update_time = self.push_update(i)
            print("network waiting time ", pull_model_time + push_update_time)
            print("local computation time ", end1 - start)
            print("fix computation time ", end2 - start)
            for p in self.model.parameters():
                p.grad = torch.zeros_like(p.grad)
            # measure elapsed time
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'train: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | time: {time: .4f}'.format(
                batch=i + 1,
                size=self.train_loader_len,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
                time=time.time() - start_time
            )
            bar.next()
            # print(bar.suffix)  # workaround for redirecting stdout to files
        time_elapsed = bar.elapsed_td
        self.sock.send(pickle.dumps("finish"))
        msg = pickle.loads(self.sock.recv())
        assert msg == 'ok'
        bar.finish()
        return losses.avg, top1.avg, top5.avg, time_elapsed

    def validate(self, start_time, epoch):
        bar = Bar('Processing', max=self.val_loader_len)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            #target = target.cuda(non_blocking=True)

            with torch.no_grad():
                # compute output
                output = self.model(input)
                loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'validate: ({batch}/{size})| Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | time: {time: .4f}'.format(
                        batch=i + 1,
                        size=self.val_loader_len,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        time=time.time() - start_time
                        )
            bar.next()
            # print(bar.suffix)  # workaround for redirecting stdout to files
        bar.finish()
        return (losses.avg, top1.avg, top5.avg,self.lr)
    def adjust_learning_rate(self, epoch, iteration, num_iter):
        
        warmup_epoch = 5 if self.args.warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = self.args.epochs * num_iter

        if self.args.lr_decay == 'step':
            self.lr = self.args.lr * (self.args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
        elif self.args.lr_decay == 'cos':
            self.lr = self.args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif self.args.lr_decay == 'linear':
            self.lr = self.args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        elif self.args.lr_decay == 'schedule':
            count = sum([1 for s in self.args.schedule if s <= epoch])
            self.lr = self.args.lr * pow(self.args.gamma, count)
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_decay))

        if epoch < warmup_epoch:
            self.lr = self.args.lr * current_iter / warmup_iter
