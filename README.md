# A toolkit for designing and simulating Superconducting Nonlinear Asymmetric Inductive eLements for superconducting qubit experiments

<img width="513" height="386" alt="snail_image" src="https://github.com/user-attachments/assets/a2afdbca-223a-472a-9097-db27f0118fc9" />

This repository provides an integrated framework to desisgn SNAILs using qiskit metal, generate Ansys HFSS files automatically to simulate the designed devices classically, and then expanding on existing framework provided pyEPR to simualte the quantum parameters of the device with additional correction terms necessary for the flux biasing of SNAILs. The goal of this project is to automate and speed up the design-simulate-prep for fabrication cycle to test superconducting qubit devices. 

Sources include [pyEPR](https://pyepr-docs.readthedocs.io/en/latest/), [qiskit metal](https://pypi.org/project/qiskit-metal/), and [the thesis by Dr. Frattini](https://bpb-us-w2.wpmucdn.com/campuspress.yale.edu/dist/2/3627/files/2023/07/Frattini_thesis.pdf). 

# Planned Structure
`demos.ipynb` contains a demonstration of designing, simulating the quantum parameters, and generating a gds file to fabricate a device. Required for this step is an npz file generated from running standard pyEPR on an Ansys HFSS file which I will be providing. Ansys HFSS is a licenced software that has student licence, but for ease I will be providing those intermediate files so you do not have to rerun the electromagnetic simulation.

`src/utils_design.py` contains utilities for designing a device given some geomtric parameters or target unharmonicity, frequency, and assymetry.

`src/utils_hfss.py` contains utilities for generating the files necessary to run a finite element Ansys HFSS simulation of the designed device.

`src/utils_snail_pyepr.py` contains utilities for treating the results of pyepr to get the necessary corrections for a snail device.

`src/utils_snail_gds.py` contains utilities for generating a gds file with the designed snail device.

# Timeline
First, I will implement the design of a SNAIl device using qiskit metal.
Next, I will implement the generation of Ansys HFSS files. 
After that, I run the Ansys HFSS simlation offiline and run pyepr.
Then, I will implement the pyEPR extention.
Lastly, I will implement the generation of gds files necessary for actually fabricating this device.

# Project members
Katerina Kargioti

