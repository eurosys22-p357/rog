from numpy.lib.histograms import _ravel_and_check_weights
from torch import tensor
from torch._C import dtype, wait
from torch.distributed.distributed_c10d import recv
import socket
import queue
import threading
import pickle
import multiprocessing as mp
import time

from torch.optim import optimizer
from utils import Bar, AverageMeter, accuracy, mkdir_p, savefig
import torch
from math import cos, pi
import torch.distributed as dist
import numpy as np
import math
import operator

MAX_RECV_SIZE=4*1024
MTA=[1,0.5,0.5,0.38197,0.31767,0.27551,0.24512,0.22191,0.203456,0.188348,0.175699]
class TCPMessageStream:
    NUM_SIZE=4
    def __init__(self,sock:socket.socket):
        self.sock=sock
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
            while True:
                while len(buffer) < self.NUM_SIZE:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msize=int.from_bytes(buffer[:self.NUM_SIZE],"big")
                buffer=buffer[self.NUM_SIZE:]
                while len(buffer)<msize:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msg =buffer[:msize]
                buffer=buffer[msize:]
                self.recv_queue.put(msg)
        self.recv_thread=threading.Thread(target=recv_task)
        self.recv_thread.start()

    def send(self,msg):
        self.send_queue.put(msg)

    def recv(self):
        return self.recv_queue.get()

class layer_unit:
    def __init__(self,tensor,mtu_number):
        self.size=tensor.size()
        self.row=[]
        unsuitable=True 
        for i in range(len(self.size)-1):
            row_length=1
            for j in range(i+1,len(self.size)):
                row_length*=self.size[j]
            if row_length<mtu_number:
                row_length=int(mtu_number/row_length)*row_length
                unsuitable=False
                break    
        if unsuitable:
            if self.size[-1]<mtu_number:
                row_length=self.size[-1]
            else:
                row_length=mtu_number
        length=tensor.numel()
        num=int(length/row_length)
        end=0
        for i in range(num):
            start=int(i*row_length)
            end=int((i+1)*row_length)
            self.row.append((start,end))
        if end!=length:
            self.row.append((end,length))
        print(self.row)

    def select_model(self,tensor,selected,versions,world_size,j):
        reshaped=tensor.view(-1)
        transmission_tensor=None
        transmission_version=None
        for i in range(len(self.row)):
            if selected[i]==True:
                tag=reshaped[self.row[i][0]:self.row[i][1]]
                this_version=torch.zeros(2+world_size)
                this_version[0]=j
                this_version[1]=i
                this_version[2:]=versions[i]
                if transmission_tensor==None:
                    transmission_tensor=tag
                    transmission_version=this_version
                else:
                    transmission_tensor=torch.cat((transmission_tensor,tag),0)
                    transmission_version=torch.cat((transmission_version,this_version),0)
        return transmission_tensor,transmission_version
    def select_gradients(self,tensor,selected,versions,j):
        reshaped=tensor.view(-1)
        transmission_tensor=None
        transmission_version=None
        for i in range(len(self.row)):
            if selected[i]==True:
                tag=reshaped[self.row[i][0]:self.row[i][1]]
                this_version=torch.zeros(3)
                this_version[0]=j
                this_version[1]=i
                this_version[2]=versions[i]
                if transmission_tensor==None:
                    transmission_tensor=tag
                    transmission_version=this_version
                else:
                    transmission_tensor=torch.cat((transmission_tensor,tag),0)
                    transmission_version=torch.cat((transmission_version,this_version),0)
        return transmission_tensor,transmission_version
    
        
class Parameter_Server:
    def __init__(self,ps_ip,ps_port,world_size,threshold,model,optimizer,communication_library,MTU):
        self.world_size=world_size
        self.model=model
        self.threshold=threshold
        
        self.gathered_weight=mp.Queue(maxsize=1000)
        self.updated_by_others=mp.Queue(maxsize=10)
        self.lock=threading.Lock()
        self.training_step=[0 for _ in range(world_size)]
        self.communication_library=communication_library
        self.mtu_number=MTU/4
        self.throught=[0 for _ in range(world_size)]

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ps_ip, ps_port))
        sock.listen(world_size)

        self.layer_info=[]
        layers=0
        for _ in self.model.parameters():
            layers+=1
        layer=0
        for p_idx,p in enumerate(self.model.parameters()):
            layer+=1
            self.layer_info.append(layer_unit(p,self.mtu_number))
            assert p.dtype==torch.float32
        self.row_states=[]
        self.row_importance=[]
        for i in range(len(self.layer_info)):
            rows=[]
            for _ in range(len(self.layer_info[i].row)):
                rows.append([0 for _ in range(world_size)])
            self.row_states.append(np.array(rows))
            self.row_importance.append(np.array([0 for _ in range(len(self.layer_info[i].row))]))

        proc=[]
        self.finish=[False for _ in range(self.world_size)]
        self.allfinish=mp.Queue(maxsize=10)
        self.isupdated=[]
        for _ in range(world_size):
            self.isupdated.append(mp.Queue(maxsize=10))
        t=threading.Thread(target=self.parameter_server_optimizer,args=(optimizer,))
        proc.append(t)
       
        for i in range(world_size):
            client_sock, _ = sock.accept()
            client_stream=TCPMessageStream(client_sock)
            t=threading.Thread(target=self.each_parameter_server,args=(client_stream,ps_ip,ps_port,i))
            proc.append(t)
        for t in proc:
            t.start()
        print("start parameter server",flush=True)
    def select_model(self,transmission_rate,iteration):
        selected=[]
        importance=np.array([])
        for i in range(len(self.layer_info)):
            importance=np.concatenate((importance,self.row_importance[i]*(-1)+iteration),0)
        threshold=sorted(importance)[int(len(importance)*transmission_rate)]
        total=0
        count=0
        for i in range(len(self.layer_info)):
            tag2=len(self.layer_info[i].row)
            total+=tag2
            tag=[False for _ in range(tag2)]
            for j in range(tag2):
                if self.row_importance[i][j]>threshold:
                    tag[j]=True
                    count+=1
            selected.append(tag)
        if transmission_rate- count/total>1e-3:
            finish=False
            for i in range(len(self.layer_info)):
                for j in range(len(self.layer_info[i].row)):
                    if self.row_importance[i][j]==threshold:
                        selected[i][j]=True
                        count+=1
                    if transmission_rate- count/total<1e-3:
                        finish=True
                        break
                if finish:
                    break 
        print(f"transmission rate:{transmission_rate}, actually:{count/total}",flush=True)          
        return selected
    def each_parameter_server(self,client_stream:TCPMessageStream,ps_ip,ps_port,rank):
        while True:
            msg=pickle.loads(client_stream.recv())
            print(msg)
            if msg=='get':
                client_stream.send(pickle.dumps((self.model,self.layer_info)))
            if msg[:3]=='thr':
                self.throught[rank]=float(msg[8:])
            if msg[:3]=='ask':
                selected=self.select_model(MTA[self.threshold]*self.throught[rank]/min(self.throught),self.training_step[rank])
                transmission_tensor=None
                transmission_versions=None
                for i,p in enumerate(self.model.parameters()):
                    selected_tensor,selected_verisons=self.layer_info[i].select_model(p,selected[i],self.row_states[i],self.world_size,i)
                    if transmission_tensor==None:
                        transmission_tensor=selected_tensor
                        transmission_versions=selected_verisons
                    else:
                        transmission_tensor=torch.cat((transmission_tensor,selected_tensor),0)
                        transmission_versions=torch.cat((transmission_versions,selected_verisons),0)
                srcrank=int(msg[3:])
                client_stream.send(pickle.dumps([transmission_versions.numel(),transmission_tensor.numel()]))
                dist.send(transmission_versions,srcrank)
                dist.send(transmission_tensor,srcrank)
            if msg[:3]=='sen':
                client_stream.send(pickle.dumps(MTA[self.threshold]*self.throught[rank]/min(self.throught)))
                with self.lock:
                    self.training_step[rank]+=1
                lr=float(msg[3:])
                selected=pickle.loads(client_stream.recv())
                weight=self.gather_gradient(selected)
                self.gathered_weight.put((weight,lr,rank))
                msg=self.isupdated[rank].get()
                assert msg=="ok"
                client_stream.send(pickle.dumps("ok"))
                while not self.updated_by_others.empty():
                    try:
                        self.updated_by_others.get_nowait()
                    except:
                        break
            if msg=="finish":
                with self.lock:
                    self.finish[rank]=True
                    condition=all(self.finish)
                if condition:
                    for _ in range(self.world_size-1):
                        self.allfinish.put("ok")
                else:
                    self.allfinish.get()
                self.training_step[rank]=0
                for i in range(len(self.layer_info)):
                    for j in range(len(self.layer_info[i].row)):
                        for k in range(self.world_size):
                            self.row_states[i][j][k]=0
                client_stream.send(pickle.dumps("ok"))
            if msg[:3]=='nee':
                length=int(msg[3:])
                recv_tensor=torch.tensor([0 for _ in range(length)])
                dist.recv(recv_tensor,srcrank)
                selected=[]
                for i in range(self.layer_info):
                    selected.append([False for _ in range(len(self.layer_info[i].row))])
                for i in range(length/3):
                    selected[recv_tensor[i*3]][recv_tensor[i*3+1]]=True
                    min_step=min(self.row_states[recv_tensor[i*3]][recv_tensor[i*3+1]])
                    while recv_tensor[3*i+2]>min_step+self.threshold:
                        self.updated_by_others.get()
                transmission_tensor=None
                transmission_versions=None
                for i,p in enumerate(self.model.parameters()):
                    selected_tensor,selected_verisons=self.layer_info[i].select_model(p,selected[i],self.row_states[i],self.world_size,i)
                    if transmission_tensor==None:
                        transmission_tensor=selected_tensor
                        transmission_versions=selected_verisons
                    else:
                        transmission_tensor=torch.cat((transmission_tensor,selected_tensor),0)
                        transmission_versions=torch.cat((transmission_versions,selected_verisons),0)
                client_stream.send(pickle.dumps([transmission_versions.numel(),transmission_tensor.numel()]))
                dist.send(transmission_versions,srcrank)
                dist.send(transmission_tensor,srcrank)



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
    
    def gather_gradient(self,selected):
        weight=[]
        srcrank=selected[0]
        recv_numbers=torch.tensor([0 for _ in range(selected[1])])
        dist.recv(recv_numbers,srcrank)
        recv_gradients=torch.tensor([0 for _ in range(selected[2])])
        dist.recv(recv_gradients,srcrank)  
        idx=0
        offset=0
        weight=[]
        for i,p in enumerate(self.model.parameters()):
            this_layer_gradients=torch.zeros_like(p).view(-1)
            while recv_numbers[idx]==i:
                row_idx=recv_numbers[idx+1]
                self.row_states[i][row_idx][srcrank]+=recv_numbers[idx+2]
                start=self.layer_info[i].rows[row_idx][0]
                end=self.layer_info[i].rows[row_idx][1]
                this_layer_gradients[start:end]=recv_gradients[offset:offset+end-start]
                self.row_importance[i][row_idx]=min(self.row_states[i][row_idx])
                offset+=end-start
                idx+=3
            weight.append(this_layer_gradients.reshape(p.size()))
        return weight

class Local_Worker:
    def __init__(self,args,model,ps_ip,ps_port,train_loader, train_loader_len,val_loader, val_loader_len, criterion,optimizer,communication_library):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1) # waiting for ps start
        sock.connect((ps_ip, ps_port))
        self.immediately_queue=mp.Queue()
        self.sock = TCPMessageStream(sock,self.immediately_queue)
        print("conneted to parameter server", flush=True)
        
        
        self.args=args
        self.model=model
        self.criterion=criterion
        self.train_loader=train_loader
        self.train_loader_len=train_loader_len
        self.val_loader=val_loader
        self.val_loader_len=val_loader_len
        self.lr=args.lr
        self.optimizer=optimizer
        self.communication_library=communication_library
        self.world_size=args.world_size-1
        self.threshold=args.threshold

        def listen_bandwidth():
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
                    self.sock.send(pickle.dumps("thr"+str((new-transmit)/interval*8/1024/1024)))
                    transmit=new
                    break
                time.sleep(interval)
        self.listen_thread=threading.Thread(listen_bandwidth)
        self.listen_thread.start()

        
    def update_model(self,recv_tensor,recv_versions):
        idx=0
        offset=0
        rank=dist.get_rank()
        for i,p in enumerate(self.model.parameters()):
            this_layer_model=p.view(-1)
            for j in range(len(self.layer_info[i])):
                self.row_states[i][j][rank]+=1
            while recv_versions[idx]==i:
                row_idx=recv_versions[idx+1]
                self.row_states[i][row_idx]=recv_versions[idx+2:idx+2+self.world_size]
                start=self.layer_info[i].rows[row_idx][0]
                end=self.layer_info[i].rows[row_idx][1]
                this_layer_model[start:end]=recv_tensor[offset:offset+end-start]
                offset+=end-start
                idx+=2+self.world_size
       
    def check_threshold(self,iteration):
        print("check for threshold") 
        waiting_for_new=[]
        threshold=self.args.threshold
        for i in range(len(self.row_states)):
            for j in range(len(self.row_states[i])):
                min_step=min(self.row_states[i][j])
                if iteration>min_step+threshold:
                    waiting_for_new.append((i,j,iteration))
        if waiting_for_new!=[]:
            self.sock.send(pickle.dumps("nee"+str(3*len(waiting_for_new))))
            send_tensor=torch.tensor([0 for _ in range(3*len(waiting_for_new))])
            for i in range(len(waiting_for_new)):
                send_tensor[3*i]=waiting_for_new[i][0]
                send_tensor[3*i+1]=waiting_for_new[i][1]
                send_tensor[3*i+2]=waiting_for_new[i][2]
            dist.send(send_tensor,0)
            recv_lengths=pickle.loads(self.sock.recv())
            recv_versions=torch.zeros(recv_lengths[0])
            recv_tensor=torch.zeros(recv_lengths[0])
            dist.recv(recv_versions,0)
            dist.recv(recv_tensor,0)
            self.update_model(recv_tensor,recv_versions)
            self.check_threshold(iteration)
        print("pass check threshold")
               
    def pull_model(self,iteration):
        print("start ask for new model",flush=True)
        self.sock.send(pickle.dumps("ask"+str(dist.get_rank())))
        start=time.time()
        self.optimizer.step()
        recv_lengths=pickle.loads(self.sock.recv())
        recv_versions=torch.zeros(recv_lengths[0])
        recv_tensor=torch.zeros(recv_lengths[0])
        dist.recv(recv_versions,0)
        dist.recv(recv_tensor,0)
        end=time.time()
        self.update_model(recv_tensor,recv_versions)
        self.check_threshold(iteration)
        self.model.train()
        print("complete",end-start,flush=True)
        return end-start
    def select_gradients(self,transmission_rate,iteration):
        for i,p in enumerate(len(self.remain)):
            this_layer_gradient=self.remain.view(-1)
            for j in range(self.layer_info[i].row):
                start=self.layer_info[i].row[j][0]
                end=self.layer_info[i].row[j][0]
                min_step=min(self.row_states[i][j])
                factor=iteration-min_step
                if iteration>min_step+self.threshold:
                    factor=float("inf")
                self.row_importance[i][j]=torch.sum(torch.abs(this_layer_gradient[self.row[i][0]:self.row[i][1]])).item()*factor

        selected=[]
        importance=np.array([])
        for i in range(len(self.layer_info)):
            importance=np.concatenate((importance,self.row_importance[i]),0)
        threshold=sorted(importance)[int(len(importance)*transmission_rate)]
        total=0
        count=0
        for i in range(len(self.layer_info)):
            tag2=len(self.layer_info[i].row)
            total+=tag2
            tag=[False for _ in range(tag2)]
            for j in range(tag2):
                if self.row_importance[i][j]>threshold:
                    tag[j]=True
                    count+=1
            selected.append(tag)
        if transmission_rate- count/total>1e-3:
            finish=False
            for i in range(len(self.layer_info)):
                for j in range(len(self.layer_info[i].row)):
                    if self.row_importance[i][j]==threshold:
                        selected[i][j]=True
                        count+=1
                    if transmission_rate- count/total<1e-3:
                        finish=True
                        break
                if finish:
                    break 
        print(f"transmission rate:{transmission_rate}, actually:{count/total}",flush=True)          
        return selected
    def push_update(self,iteration):
        print("start push update",flush=True)
        start=time.time()
        self.sock.send(pickle.dumps("sen"+str(self.lr)))
        transmission_rate=pickle.loads(self.sock.recv())
        selected=self.select_gradients(transmission_rate)
        transmission_gradients=None
        transmission_numbers=None
        for i,p in enumerate(self.model.parameters()):
            selected_gradients,selected_numbers=self.layer_info[i].select_gradients(p,selected[i],self.remain_number[i],i)
            if transmission_gradients==None:
                transmission_gradients=selected_gradients
                transmission_numbers=selected_numbers
            else:
                transmission_gradients=torch.cat((transmission_gradients,selected_gradients),0)
                transmission_numbers=torch.cat((transmission_numbers,selected_numbers),0)
        self.sock.send(pickle.dumps([dist.get_rank(),transmission_numbers.numel(),transmission_gradients.numel()]))
        dist.send(transmission_numbers,0)
        dist.send(transmission_gradients,0)
        end = time.time()
        for i in range(len(selected)):
            for j in range(len(selected[i][j])):
                if selected[i][j]==False:
                    continue
                p=self.remain[i]
                p.data[self.layer_info[i].row[j][0]:self.layer_info[i].row[j][1]]=torch.zeros_like(p[self.layer_info[i].row[j][0]:self.layer_info[i].row[j][1]])
                self.remain_number[i][j]=0
                self.row_importance[i][j]=0
        print("complete",flush=True)
        return end - start
        
    def get_model(self):
        self.sock.send(pickle.dumps("get"))
        self.model,self.layer_info=pickle.loads(self.sock.recv())
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
        self.row_importance=[]
        for i,p in enumerate(self.model.parameters()):
            self.remain.append(torch.zeros_like(p))
            self.remain_number.append(np.array([0 for _ in range(len(self.layer_info[i].row))]))
            self.row_importance.append(np.array([0 for _ in range(len(self.layer_info[i].row))]))
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
            self.optimizer.zero_grad()
            loss.backward()
            end1 = time.time()
            if (self.args.fix-end1+start>0):
                time.sleep(self.args.fix-end1+start)
            end2=time.time()
            push_update_time = self.push_update(i)
            print("network waiting time ", pull_model_time + push_update_time)
            print("local computation time ", end1 - start)
            print("fix computation time ", end2 - start)
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
