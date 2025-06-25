import numpy as np

import matplotlib.pyplot as plt

loss_vlasov = np.load("loss_vlasov.npy")

loss_tanh = np.load("loss_tanh.npy")

print(loss_vlasov)


plt.semilogy(loss_vlasov, label="vlasov sheath")
plt.semilogy(loss_tanh, label="tanh sheath")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
