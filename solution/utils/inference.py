import numpy as np
import pandas as pd
from .utils import softmax


def basic_inference(test_dataset, trainer, task_infos, training_args):
    logits = trainer.predict(test_dataset)[0]
    probs = softmax(logits).tolist()
    result = np.argmax(logits, axis=-1).tolist()
    pred_answer = [task_infos.id2label[v] for v in result]
    
    return probs, pred_answer    
    
INFERENCE_PIPELINE = {
    "basic": basic_inference,
}