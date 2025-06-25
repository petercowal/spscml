import sys
sys.path.append("src")
sys.path.append("tesseracts")
import math
import json
import random

import numpy as np

from tesseract_core import Tesseract

import sheaths.tanh_sheath.tesseract_api as tanh_sheath_tesseract_api


for iter in range(200):

    #inputs = {"n": 6e22, "T": 20.0, "Vp": 1500.0, "Lz": 0.5 }
    inputs = {"n": 10**(random.uniform(18,28)), "T": 2**random.uniform(-1,14), "Vp": random.uniform(200,14000), "Lz": 0.5}

    with Tesseract.from_tesseract_api(tanh_sheath_tesseract_api) as sheath_tx:
        result = sheath_tx.apply(inputs)
        if not np.isnan(result["j"]):
            jacobian = sheath_tx.jacobian(inputs, jac_inputs = ["n","T","Vp"], jac_outputs = ["j"])
            data_to_write = {"input": inputs, "output": result, "jacobian": jacobian}

            with open("training_data_tanh.txt", "a") as myfile:
                myfile.write(json.dumps(data_to_write))
                myfile.write('\n')
