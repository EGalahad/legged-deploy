import onnxruntime as ort
import json
import numpy as np
from typing import Dict


class ONNXModule:
    
    def __init__(self, path: str):

        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.meta = json.load(path.replace(".pt", ".json"), "r")
        self.in_keys = self.meta["in_keys"]
        self.out_keys = self.meta["out_keys"]
    
    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key] 
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.meta["out_keys"], outputs)}
        return outputs
    
