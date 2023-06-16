import os, sys
from datetime import datetime


class MyLogger:
    _INSTANCES = dict()
    """
        This class is a logger that can print both to a file and stdout without redirecting the output of the script
        Call MyLogger.setup('name', 'path') anywhere in your program and then you can use MyLogger.get('name').log('message') in
        any other part of the program.
    """
    def __init__(self, name, path, log_time=True):
        self.name = name
        self.log_time = log_time
        self.file = open(path, 'w')

        if self.log_time:
            self.start_time = datetime.now()
            self.log(f'Started at {self.start_time}')
    
    def log(self, message='', end='\n', log_console=True, log_file=True, log_error=False):
        text = f'[{self.name}] {message}'
        if log_error:
            print(text, end=end, file=sys.stderr)
        if log_console:
            print(text, end=end)
            sys.stdout.flush()
        if log_file:
            self.file.write(f'{text}{end}')
            self.file.flush()
        return self

    def close(self):
        if self.log_time:
            end_time = datetime.now()
            self.log(f'Ended at {end_time}')
            self.log(f'Elapsed: {end_time - self.start_time}')
        self.file.close()

    @staticmethod
    def setup(name, path):
        if MyLogger._INSTANCES.get(name, None) is not None: # already exists
            MyLogger._INSTANCES[name].close() # safely close it
        MyLogger._INSTANCES[name] = MyLogger(name, path) # create a new instance
        return MyLogger._INSTANCES[name]

    @staticmethod
    def get(name):
        if MyLogger._INSTANCES.get(name, None) is None:
            raise RuntimeError(f'Instance does not exist, please call setup(...) first!')
        return MyLogger._INSTANCES[name]

    @staticmethod
    def destroy(name):
        if MyLogger._INSTANCES.get(name, None) is not None:
            MyLogger._INSTANCES[name].close()
