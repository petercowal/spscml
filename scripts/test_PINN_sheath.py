import sys
sys.path.append("src")

import matplotlib.pyplot as plt

from spscml.pinn_vlasov.sheath_model import calculate_plasma_current


results = calculate_plasma_current(1500, 20, 6e22, 0.5)

print(results)
