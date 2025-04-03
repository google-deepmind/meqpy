# README
MEQPY is a python interface to the MEQ code that runs using Octave.
MEQ is a suite of codes for tokamak magnetic equilibrium calculations developed
by SPC-EPFL.

## Installation

To install and run MEQPY we have to do a few steps first:

* Install Octave:
```
sudo apt-get install octave
sudo apt-get install octave-dev
```

* Download the latest version of MEQ:

See the repository instructions

* Build MEQ using Octave following the instructions in the MEQ repo. This will
create a `genlib` directory in the current directory.

* Create a Python3 virtual environment:
```
sudo apt-get install python3-tk
sudo apt install python3.11-venv

python3 -m venv meqpyvenv
```

* Set the `MAT_ROOT` environment variable to the path where you have installed
MEQ and any other mat files (for example if MEQ is installed in `$HOME/meq`)
then the path to MEQ would be `$HOME`. Any other mat files should be passed
using the `extra_paths` argument in the `MeqPy` constructor.

The path to the files can also be set in the constructor of `MeqPy` using
`root_dir` but for testing the environment variable is easier.

```
export MAT_ROOT=<path to MEQ>
```

* Clone this package (using the command above) and install it using:
```
cd meqpy
pip install -e .
```

* To run the tests
```
pytest
```

## Minimal Examples using Anamak
Anamak is an analytically-defined tokamak used for tests inside MEQ.

FGE is the free-boundary evolution solver used in MEQ.

Here is a minimal example to initialize an FGE environment and run a simulation.

```python

import meqpy
import numpy as np
import matplotlib.pyplot as plt

tokamak = "ana"
time = 0 # initial time
source = meqpy_impl.MeqSource.MEQ_DIRECT
shot = 2; # shot number 2 is a diverted shot
cde = 'Ohmtor0D_rigid' # current diffusion equation used for simulation

# `control_timestep` % `simulator_timestep` must be 0 for the simulation to run.
control_timestep = 1e-4 # How often a new control signal is sent to the environment
simulator_timestep = 1e-4 # How often to run the simulation and evolve the simulation state

meq_instance = meqpy_impl.MeqPy()
# Setting debug=2 for help debugging (including for the initialization)
meq_instance.init_fge(
      tokamak, shot, time, source, default_meq_params={'debug': 2}, cde=cde
  )

original_bp = meq_instance.get_fge_input('bpD')
meq_instance.init_fgetk_environment(
    control_timestep=control_timestep,
    simulator_timestep=simulator_timestep,
    input_overrides={'bpD': original_bp * 1.001})

# define actions (voltages)
num_actions = len(meq_instance.get_fgetk_circuit_names())
actions = np.full((num_actions,), 10)

# get the r and z coordinates of the flux grid
rx = meq_instance.get_fge_geometry('rx').reshape(-1)
zx = meq_instance.get_fge_geometry('zx').reshape(-1)

for i in range(3):
  # run one step of the environment
  LY, stop = meq_instance.step_fgetk_environment(actions, {})

  # plot and display
  print("%d, Ip: %3.0f, FB=%6.4f, zA=%6.4f" % (i, LY.Ip, LY.FB[0], LY.zA[0]))
  plt.contour(rx,zx,LY.Fx-LY.FB,21)
  plt.contour(rx,zx,LY.Fx,LY.FB,colors='k')

  plt.axis('equal')
  plt.show()
```

## Copyright

Copyright 2025 Google LLC
All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
