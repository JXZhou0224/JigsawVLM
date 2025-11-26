class BaseTask:
    def format_input(self,case):
       raise NotImplementedError
    
    def format_inputs(self,cases):
        ret = []
        for case in cases:
            ret.append(self.format_input(case))
        return ret

    def checker(self,case,completion):
        raise Exception("not implemented in subclass")

    def check(self,cases,completions):
        ret = []
        for case,completion in zip(cases,completions):
            ret.append(self.checker(case,completion))
        return ret    

class TaskFactory:   
    _registry: dict[str,BaseTask] = {}

    @classmethod   
    def register(cls,task_class_name):
        def wrapper(task_calss):
            cls._registry[task_class_name] = task_calss
            return task_calss
        return wrapper
    
    @classmethod
    def get_task(self,task_class,**kwargs) -> BaseTask:
        if task_class not in self._registry:
            raise ValueError(f"task {task_class} is not registered, supported tasks:\n"+str(self._registry.keys()))
        return self._registry[task_class](**kwargs)
    
