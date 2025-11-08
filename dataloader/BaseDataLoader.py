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
    
    def checker(self,case,completion):
        raise Exception("not implemented in subclass")

    def check(self,cases,completions):
        ret = []
        for case,completion in zip(cases,completions):
            ret.append(self.checker(case,completion))
        return ret    

    def get_samples(i):
        raise Exception("not implemented in subclass")

    def __iter__(self):
        for i in range(self.n):
            yield self.get_samples(i)
    