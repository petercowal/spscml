import numpy as np

from tesseract_core import Tesseract

inputs = {"n": 6e22, "T": 20.0, "Vp": 1500.0, "Lz": 0.5 }


with Tesseract.from_image("vlasov_sheath") as sheath_tx:
    result = sheath_tx.apply(inputs)

print(result)
