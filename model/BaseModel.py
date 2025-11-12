class BaseModel:
    def generate(self,messages):
        pass

class ModelFactory:   
    _registry: dict[str,BaseModel] = {}

    @classmethod   
    def register(cls,model_class_name):
        def wrapper(llm_class):
            cls._registry[model_class_name] = llm_class
            return llm_class
        return wrapper
    
    @classmethod
    def get_llm(self,model_class,**kwargs) -> BaseModel:
        if model_class not in self._registry:
            raise ValueError(f"Model {model_class} is not registered, supported models:\n"+str(self._registry.keys()))
        return self._registry[model_class](**kwargs)
    
