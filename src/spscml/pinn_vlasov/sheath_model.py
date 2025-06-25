import numpy as np
import jax
import jax.numpy as jnp

import jpu


def calculate_plasma_current(Vp, T, n, Lz, **kwargs):
    '''
    Calculates the plasma current carried by a plasma with the given temperature and
    number density across an electrode gap at the given voltage.

    Args:
        - Vp: The electrode gap voltage [volts]
        - T: The plasma temperature [eV]
        - n: The plasma volumetric number density [m^-3]

    Returns: a dictionary containing solution values at the final time
        - fe: The electron distribution function
        - fi: The ion distribution function
        - electron_grid: The phase space grid for the electrons
        - ion_grid: The phase space grid for the ions
        - je: The electron current density [amperes / m^2]
        - ji: The ion current density [amperes / m^2]
        - javg: The space-averaged total current density [amperes / m^2]
        - E: The electric field [volts/meter]
        - ne: The electron density
        - ni: The ion density
    '''
    #jax.debug.print("=====================================================")
    #jax.debug.print("Running a PINN sheath model with the following parameters:")
    #jax.debug.print("    Voltage: {} [volts]", Vp)
    #jax.debug.print("    Density: {} [meters^-3]", n)
    #jax.debug.print("    Temperature: {} [electron-volts]", T)
    jax.debug.print("Voltage: {} [volts]", Vp)
    pinn_inputs = jnp.array([n, T, Vp])

    # load the neural network
    np_weights = np.load("model_weights.npz")
    linear1 = np_weights["arr_0"]
    bias1 = np_weights["arr_1"]
    linear2 = np_weights["arr_2"]
    bias2 = np_weights["arr_3"]
    linear3 = np_weights["arr_4"]
    bias3 = np_weights["arr_5"]
    linear4 = np_weights["arr_6"]
    bias4 = np_weights["arr_7"]

    x1 = linear1 @ jnp.log(pinn_inputs)
    x2 = linear2 @ jnp.tanh(x1 + bias1)
    x3 = linear3 @ jnp.tanh(x2 + bias2)
    x4 = linear4 @ jnp.tanh(x3 + bias3)
    j_avg = -jnp.exp(x4 + bias4)

    #jax.debug.print("")
    #jax.debug.print("Result of the PINN simulation:")
    #jax.debug.print("    Average current density j = {}, [amperes / meters^2]", j_avg)
    #jax.debug.print("")
    #jax.debug.print("")

    return dict(j_avg=j_avg[0])
