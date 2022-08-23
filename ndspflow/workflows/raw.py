"""Read arrays from raw files."""



class Raw:

    def __init__(self):
        pass

    def read_raw(self, func, raw_iter, *args, **kwargs):

        self.y_array = raw_iter

        self.nodes.append(['read_raw', func, raw_iter, args, kwargs])

    def run_read_raw(self, func, raw_iter, args, kwargs):

        self.y_array = func(raw_iter, *args, **kwargs)