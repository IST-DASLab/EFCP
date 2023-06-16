import os
import sys
import socket
import time
from multiprocessing import Process

from helpers.mylogger import MyLogger
from helpers.tools import write_yaml

from helpers.tools import on_windows
if not on_windows():
    import gpustat

PORT = 5005
TIMEOUT_SOCKET = 0.01 # seconds
TIMEOUT_READS = 0.5
BUFFER_SIZE = 3 # bytes for message


class Message:
    STOP = 'end'
    EPOCH = 'epc'
    READY = 'rdy'
    # STOP        = 'proc___stop'
    # TRAIN_START = 'train_start'
    # TRAIN_END   = 'train___end'
    # TEST_START  = 'test__start'
    # TEST_END    = 'test____end'


def read_gpu_mem(gid, pid):
    try:
        gpu_stat = gpustat.new_query().gpus
        for proc in gpu_stat[gid]['processes']:
            if proc['pid'] == pid:
                return proc['gpu_memory_usage']
    except:
        return None


class GPUServer:
    _INSTANCE = None

    def __init__(self):
        home = os.path.expanduser('~')
        MyLogger.setup(name='gpu-server', path=os.path.join(home, 'gpu', 'gpu-server.txt'))
        self.server_socket = socket.socket()
        self.server_socket.bind(('', PORT))
        self.server_socket.listen(2)
        self.client_socket, self.client_address = self.server_socket.accept()
        MyLogger.get('gpu-server').log(f'Client #{self.client_address}# connected from #{self.client_socket}#')

    def send(self, data):
        self.client_socket.send(data.encode())
        MyLogger.get('gpu-server').log(f'Sent #{data}# to client')

    def close(self):
        MyLogger.get('gpu-server').log('Closing client & server sockets')
        self.client_socket.close()
        self.server_socket.close()

    @staticmethod
    def get():
        if GPUServer._INSTANCE is None:
            GPUServer._INSTANCE = GPUServer()
        return GPUServer._INSTANCE

    @staticmethod
    def destroy():
        GPUServer._INSTANCE.close()
        del GPUServer._INSTANCE
        GPUServer._INSTANCE = None


def gpu_client(args, gid, pid, file):
    """
        This method monitors the memory used by the process `pid` on the GPU `gid`
        :param args: the dictionary from argparse
        :param gid: the GPU id to be monitored
        :param pid: the process id to be monitored
        :param file:
        :return: nothing, but writes the memory usage history in `file`
    """
    MyLogger.setup(name='gpu-client', path=os.path.join(args.root_folder, 'gpu-client.txt'))
    MyLogger.get('gpu-client').log('GPUClient is sleeping')
    time.sleep(5)
    sock = socket.socket()
    sock.connect(('localhost', PORT))
    sock.settimeout(TIMEOUT_SOCKET)


    mem_reads = []
    now = time.time()
    time.sleep(1)
    while True:
        if time.time() - now > TIMEOUT_READS:
            mem = read_gpu_mem(gid, pid)
            if mem is not None:
                MyLogger.get('gpu-client').log(str(mem))
                mem_reads.append(mem)
            now = time.time()
        try:
            data = sock.recv(BUFFER_SIZE).decode()
            if data is not None and len(data) > 0:
                MyLogger.get('gpu-client').log(f'received: #{data}#')
                if data == Message.STOP:
                    break
                elif data == Message.EPOCH:
                    mem_reads.append(None)
        except socket.timeout:
            pass
    MyLogger.get('gpu-client').log(f'Writing to file {file}')

    write_yaml(file, data=dict(description=args.wandb_job_type, mem_reads=str(mem_reads)))
    MyLogger.get('gpu-client').log('GPU monitoring ended')
    MyLogger.destroy('gpu-client')

    sock.send(Message.READY.encode())
    sock.close()


def start_gpu_monitoring(args):
    """
        This method does the following:
        - starts the server (initializes the instance in the class)
        - starts the client on another process
    """
    gid = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    pid = os.getpid()
    file = os.path.join(args.root_folder, 'mem_reads.txt')

    p = Process(target=gpu_client, args=(args, gid, pid, file))
    p.start()
    GPUServer.get() # initialize the server
    return p
