from datasets import Dataset
class BaseDataLoader():
    def __init__(self,**kwargs): 
        self.n = None
        self.batch_size = None
        self.data:Dataset = None
        self.load_data(**kwargs)
        assert self.n!= None and self.batch_size != None and self.data !=None,"init error"
    
    def load_data(self,**kwargs):
        raise Exception("not implemented in subclass") 

    def get_samples(i):
        raise Exception("not implemented in subclass")

    def __iter__(self):
        for i in range(self.n):
            yield self.get_samples(i)
    