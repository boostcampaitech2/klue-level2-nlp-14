import numpy as np
import pandas as pd
from .utils import softmax


def basic_inference(test_dataset, trainer, task_infos, training_args):
    test_id = test_dataset["guid"]
    logits = trainer.predict(test_dataset)[0]
    probs = softmax(logits).tolist()
    result = np.argmax(logits, axis=-1).tolist()
    pred_answer = [task_infos.id2label[v] for v in result]

    output = pd.DataFrame(
        {
            'id':test_id,
            'pred_label':pred_answer,
            'probs':probs,
        }
    )
    submir_dir = training_args.output_dir
    run_name = training_args.run_name
    output.to_csv(f'{submir_dir}/submission_{run_name}.csv', index=False)
    
    
INFERENCE_PIPELINE = {
    "basic": basic_inference,
}