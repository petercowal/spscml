import sys
sys.path.append("src")
sys.path.append("tesseracts")

import numpy as np

from tesseract_core import Tesseract

import sheaths.vlasov.tesseract_api as vlasov_sheath_tesseract_api
import sheaths.pinn_sheath.tesseract_api as pinn_sheath_tesseract_api
inputs = {"n": 6e22, "T": 20.0, "Vp": 1500.0, "Lz": 0.5 }


with Tesseract.from_tesseract_api(vlasov_sheath_tesseract_api) as sheath_tx:
    result = sheath_tx.apply(inputs)
    #jacobian = sheath_tx.jacobian(inputs, jac_inputs = ["n","T","Vp"], jac_outputs = ["j"])



print(result["j"])
