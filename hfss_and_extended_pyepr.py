# ============================================================
# Imports
# ============================================================

import numpy as np
import math as m
import matplotlib.pyplot as plt
from sympy import *
import sympy as sym 
import pandas as pd
import logging
from scipy.optimize import minimize_scalar
import os

import pyEPR as epr
import numbers
from pyaedt import Desktop, Hfss
from scipy.special import factorial

# ============================================================
# Generate ansys hfss geomtry
# ============================================================


class hfss_reinhold:
    """
    HFSS geometry-builder for a 3D snail with a post cavity.

    This class constructs a chip, readout resonator, junction pads, 
    vacuum region, coupling pins, and solution setup inside HFSS 
    using PyAEDT. It also exports all numeric parameters to the HFSS
    design and creates mesh assignments.

    Parameters
    ----------
    project_name : str
        Name of the HFSS project file.
    design_name : str
        Name of the HFSS design.
    setup_names : list[str]
        Names of solution setups (only the first is used here).

    chip_w, chip_l, chip_t : float
        Chip width, length, and thickness (mm).

    res_length, res_width, res_start_y : float
        Geometric parameters of the readout resonator trace (mm).

    res_pad_dist, pad_length, pad_width : float
        Pad spacing and pad geometry (mm).

    jun_length, jun_width : float
        Dimensions of the Josephson junction geometry footprint (mm).

    Lj : float
        Junction inductance (nH).

    clamp_l : float
        Length of the clamp region (mm).

    chip_tun_r, chip_tun_l : float
        Chip tunnel radius and length (mm).

    p_cav_* : float
        Geometry of the post cavity tunnel and post.

    r_pin_*, q_pin_*, s_pin_* : float
        Geometry of readout, qubit, and storage pins..
    """
    def __init__(self,project_name, design_name, setup_names, chip_w = 5,  chip_l = 41.05, chip_t = 0.725, res_length = 8.86, res_width = 0.155, res_start_y = 28.46505,
                  res_pad_dist = 0.85, pad_length = 1, pad_width =  0.4, jun_length = 0.15, jun_width = 0.01, Lj = 7, clamp_l = 5, 
                  chip_tun_r = 3.5, chip_tun_l = 37, p_cav_tunnel_y = 45.722, p_cav_tunnel_z = -9.8317, p_cav_tun_l = 39.878, 
                  p_cav_tun_r = 5.3721, p_cav_post_l = 12.573, p_cav_post_r = 1.9939, r_pin_x = 15.722, r_pin_y = 25.722,
                  r_pin_z = 2.5, r_pin_r = 0.65, r_pin_l = -13, r_pin_tun_x = None, r_pin_tun_r = 1.18745, r_pin_tun_l = -12,
                  q_pin_x = 0.215, q_pin_y = 32.722, q_pin_z = -9.2662, q_pin_r = 0.65, q_pin_l = 9, q_pin_tun_z = None,
                  q_pin_tun_r = 1.18745, q_pin_tun_l = 9, s_pin_x = 15.4136, s_pin_y = 45.722, s_pin_z = 12.693, s_pin_r = 0.65, 
                  s_pin_l = -11, s_pin_tun_x = None, s_pin_tun_r = 1.18745, s_pin_tun_l = -11):
        
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

        self.res_start_z = (self.chip_w-self.res_width)/2 #in mm
        self.left_pad_start_y = self.res_start_y + self.res_length + self.res_pad_dist
        self.left_pad_start_z = (self.chip_w-self.pad_length)/2 #in mm
        self.junction_start_y = self.res_start_y + self.res_length + self.res_pad_dist + self.pad_width
        self.junction_start_z = (self.chip_w-self.jun_width)/2 #in mm
        self.right_pad_start_y = self.res_start_y + self.res_length + self.res_pad_dist + self.pad_width + self.jun_length
        self.right_pad_start_z = (self.chip_w-self.pad_length)/2 #in mm could adjust to have assymetric pads
        self.chip_tunnel_x = self.chip_t/2
        self.chip_tunnel_z = self.chip_w/2
        self.p_cav_tunnel_x = self.chip_t/2

         # default derived parameters
        if self.r_pin_tun_x is None:
            self.r_pin_tun_x = self.r_pin_x - 1
        if self.q_pin_tun_z is None:
            self.q_pin_tun_z = self.q_pin_z + 1
        if self.s_pin_tun_x is None:
            self.s_pin_tun_x = self.s_pin_x - 1
        
    def start_hfss_file(self):
        """
        Connect to HFSS and load the project/design.

        Initializes:
            self.hfss : pyaedt.Hfss
                Active HFSS design object.

        Notes
        -----
        - Uses solution type `"Eigenmode"`.
        - Requires an active ANSYS Electronics Desktop licence 
          and automatically opens a desktop session.
        """
        self.hfss = Hfss(
        project=self.project_name, 
        design=self.design_name, 
        solution_type="Eigenmode")

    def export_numeric_parameters(self):
        """
        Export all numeric instance attributes to HFSS as design variables.

        Notes
        -----
        - All numeric attributes are exported as `"value mm"` except `Lj`,
          which is exported as `"value nH"`.
        - Uses direct variable assignment to ``self.hfss[name]``.
        """

        for name, value in self.__dict__.items():
            if isinstance(value, numbers.Number):
                if name == "Lj":
                    # Convert Henries -> nanohenries for HFSS
                    self.hfss[name] = f"{value}nH"
                else:
                    self.hfss[name] = f"{value}mm"

    def readout_resonator(self):
        """
        Create and mesh the readout resonator strip.

        Creates
        -------
        - A rectangle forming the resonator trace.
        - Perfect electric boundary.
        - A local mesh refinement region.
        """
        self.hfss.modeler.create_rectangle(origin=["chip_t", "res_start_y", "res_start_z"], sizes=["res_length", "res_width"], orientation="YZ", name="Readout_resonator")
        self.hfss.assign_perfecte_to_sheets(assignment=["Readout_resonator"], name="Readout_resonator_perf_e")
        self.hfss.mesh.assign_length_mesh(assignment=["Readout_resonator"],  maximum_length="0.05mm", inside_selection=True, maximum_elements=None, name="Readout_resonator_mesh")
    
    def junction_pads(self):
        """
        Create the pads and Josephson junction.

        Creates
        -------
        - Left pad (PEC).
        - Right pad (PEC).
        - JJ rectangle with a Parallel lumped RLC inductance boundary.
        - Polyline defining the JJ current direction.
        - Mesh refinements for pads and junction.

        Notes
        -----
        Junction inductance uses the value exported under variable ``Lj``.
        """
        self.hfss.modeler.create_rectangle(origin=["chip_t", "left_pad_start_y", "left_pad_start_z"], sizes=["pad_width", "pad_length"], orientation="YZ", name="Left_pad")
        self.hfss.assign_perfecte_to_sheets(assignment=["Left_pad"], name="Left_pad_perf_e")

        self.hfss.modeler.create_rectangle(origin=["chip_t", "junction_start_y", "junction_start_z"], sizes=["jun_length", "jun_width"], orientation="YZ", name="Junction")
        self.hfss.assign_lumped_rlc_to_sheet(assignment="Junction", start_direction=[[self.chip_t, self.junction_start_y, self.junction_start_z + self.jun_width/2],
            [self.chip_t, self.junction_start_y + self.jun_length, self.junction_start_z + self.jun_width/2]], 
            rlc_type='Parallel', resistance=None, inductance=self.Lj, capacitance=None,  name="Junction_boundary")
        self.hfss.modeler.create_polyline([[self.chip_t, self.junction_start_y, self.junction_start_z + self.jun_width/2],
            [self.chip_t, self.junction_start_y + self.jun_length, self.junction_start_z + self.jun_width/2]],
            segment_type=None, cover_surface=False, close_surface=False, name="Junction_polyline", material=None,non_model=True)

        self.hfss.modeler.create_rectangle(origin=["chip_t", "right_pad_start_y", "right_pad_start_z"], sizes=["pad_width", "pad_length"], orientation="YZ", name="Right_pad")
        self.hfss.assign_perfecte_to_sheets(assignment=["Right_pad"], name="Right_pad_perf_e")

        self.hfss.mesh.assign_length_mesh(assignment=["Left_pad"],  maximum_length="0.15mm", inside_selection=True, maximum_elements=None, name="Left_pad_mesh")
        self.hfss.mesh.assign_length_mesh(assignment=["Right_pad"],  maximum_length="0.15mm", inside_selection=True, maximum_elements=None, name="Right_pad_mesh")
        self.hfss.mesh.assign_length_mesh(assignment=["Junction"],  maximum_length="0.001mm", inside_selection=True, maximum_elements=None, name="Junction_mesh")


    def readout_pin(self):
        """
        Create the readout coupling pin and its tunnel in the package.

        Adds:
        -------
        - PEC cylinders for the main pin and tunnel.
        - A circular boundary sheet with a 50 Ω lumped port.
        """

        self.hfss.modeler.create_cylinder(orientation="X", origin=["r_pin_x", "r_pin_y", "r_pin_z"], radius="r_pin_r", height="r_pin_l", name="Readout_pin", material="pec")
        self.hfss.modeler.create_cylinder(orientation="X", origin=["r_pin_tun_x", "r_pin_y", "r_pin_z"], radius="r_pin_tun_r", height="r_pin_tun_l", name="Readout_pin_tunnel", material="pec")
        self.hfss.modeler.create_circle(origin=[self.r_pin_tun_x, self.r_pin_y, self.r_pin_z], radius=self.r_pin_tun_r, num_sides=0, orientation="YZ", is_covered=True, name="Readout_boundary")
        self.hfss.modeler.subtract("Readout_boundary","Readout_pin", keep_originals=True) 
        self.hfss.assign_lumped_rlc_to_sheet(assignment="Readout_boundary", start_direction=[[self.r_pin_tun_x, self.r_pin_y+self.r_pin_r, self.r_pin_z],
            [self.r_pin_tun_x, self.r_pin_y+self.r_pin_tun_r, self.r_pin_z]], rlc_type='Parallel', resistance=50, 
            inductance=None, capacitance=None,  name="Readout_pin_boundary")

    def qubit_pin(self):
        """
        Create the qubit coupling pin and its tunnel in the package.

        Adds:
        -------
        - PEC cylinders for the main pin and tunnel.
        - A circular boundary sheet with a 50 Ω lumped port.
        """
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["q_pin_x", "q_pin_y", "q_pin_z"], radius="q_pin_r", height="q_pin_l", name="Qubit_pin", material="pec")
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["q_pin_x", "q_pin_y", "q_pin_tun_z"], radius="q_pin_tun_r", height="q_pin_tun_l", name="Qubit_pin_tunnel", material="pec")
        self.hfss.modeler.create_circle(origin=[self.q_pin_x, self.q_pin_y, self.q_pin_tun_z], radius=self.q_pin_tun_r, num_sides=0, orientation="XY", is_covered=True, name="Qubit_boundary")
        self.hfss.modeler.subtract("Qubit_boundary","Qubit_pin", keep_originals=True) 
        self.hfss.assign_lumped_rlc_to_sheet(assignment="Qubit_boundary", start_direction=[[self.q_pin_x, self.q_pin_y+self.q_pin_r, self.q_pin_tun_z],
            [self.q_pin_x, self.q_pin_y+self.q_pin_tun_r, self.q_pin_tun_z]], rlc_type='Parallel', resistance=50, inductance=None, capacitance=None,  name="Qubit_pin_boundary")

    def storage_pin(self):
        """
        Create the storage coupling pin and its tunnel in the package.

        Adds:
        -------
        - PEC cylinders for the main pin and tunnel.
        - A circular boundary sheet with a 50 Ω lumped port.
        """
        self.hfss.modeler.create_cylinder(orientation="X", origin=["s_pin_x", "s_pin_y", "s_pin_z"], radius="s_pin_r", height="s_pin_l", name="Storage_pin", material="pec")
        self.hfss.modeler.create_cylinder(orientation="X", origin=["s_pin_tun_x", "s_pin_y", "s_pin_z"], radius="s_pin_tun_r", height="s_pin_tun_l", name="Storage_pin_tunnel", material="pec")
        self.hfss.modeler.create_circle(origin=[self.s_pin_tun_x, self.s_pin_y, self.s_pin_z], radius=self.s_pin_tun_r, num_sides=0, orientation="YZ", is_covered=True, name="Storage_boundary")
        self.hfss.modeler.subtract("Storage_boundary","Storage_pin", keep_originals=True) 
        self.hfss.assign_lumped_rlc_to_sheet(assignment="Storage_boundary", start_direction=[[self.s_pin_tun_x, self.s_pin_y+self.s_pin_r, self.s_pin_z],
            [self.s_pin_tun_x, self.s_pin_y+self.s_pin_tun_r, self.s_pin_z]], rlc_type='Parallel', resistance=50, inductance=None, capacitance=None,  name="Storage_pin_boundary")

    def vacuum(self):
        """
        Create the global vacuum region inside the chip tunnel and package.

        Creates
        -------
        - Chip bounding box (vacuum fill).
        - Chip tunnel.
        - P-cavity tunnel.
        - Readout, qubit, and storage pin vacuum volumes.

        Notes
        -----
        - Subtracts chip and pin volumes from the tunnel.
        - Ensures correct boundary hierarchy.
        """
        self.hfss.modeler.create_box([0, 0, 0], ["chip_t", "chip_l", "chip_w"], name="C_Chip", material="vacuum")
        #self.hfss.modeler.create_box([0, 0, 0], ["chip_t", "clamp_l", "chip_w"], name="Clamp", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="Y", origin=["chip_tunnel_x", "clamp_l", "chip_tunnel_z"], radius="chip_tun_r", height="chip_tun_l", name="Chip_tunnel", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["p_cav_tunnel_x", "p_cav_tunnel_y", "p_cav_tunnel_z"], radius="p_cav_tun_r", height="p_cav_tun_l", name="P_cavity_tunnel", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="X", origin=["r_pin_x", "r_pin_y", "r_pin_z"], radius="r_pin_r", height="r_pin_l", name="C_Readout_pin", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["q_pin_x", "q_pin_y", "q_pin_z"], radius="q_pin_r", height="q_pin_l", name="C_Qubit_pin", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="X", origin=["s_pin_x", "s_pin_y", "s_pin_z"], radius="s_pin_r", height="s_pin_l", name="C_Storage_pin", material="vacuum")
        self.hfss.modeler.unite(["Chip_tunnel","P_cavity_tunnel","Storage_pin_tunnel", "Qubit_pin_tunnel", "Readout_pin_tunnel","C_Readout_pin","C_Qubit_pin","C_Storage_pin"])
        self.hfss.modeler.create_cylinder(orientation="X", origin=["r_pin_x", "r_pin_y", "r_pin_z"], radius="r_pin_r", height="r_pin_l", name="C_Readout_pin_2", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["q_pin_x", "q_pin_y", "q_pin_z"], radius="q_pin_r", height="q_pin_l", name="C_Qubit_pin_2", material="vacuum")
        self.hfss.modeler.create_cylinder(orientation="X", origin=["s_pin_x", "s_pin_y", "s_pin_z"], radius="s_pin_r", height="s_pin_l", name="C_Storage_pin_2", material="vacuum")
        self.hfss.modeler.subtract("Chip_tunnel","C_Qubit_pin_2", keep_originals=False)
        self.hfss.modeler.subtract("Chip_tunnel","C_Readout_pin_2", keep_originals=False)
        self.hfss.modeler.subtract("Chip_tunnel","C_Storage_pin_2", keep_originals=False)
        self.hfss.modeler.subtract("Chip_tunnel","C_Chip", keep_originals=False)
        #self.hfss.modeler.unite(["Chip_tunnel","Clamp"]) 

    def solution_setup(self, num_modes=4, start_f=1, max_delta=1, max_passes=10, real=True, min_pass=1, min_c_pass=2):
        """
        Create and configure an eigenmode solution setup in HFSS.

        Parameters
        ----------
        num_modes : int
            Number of eigenmodes to solve for. Default is 4.
        start_f : float
            Minimum (starting) frequency in GHz. Default is 1 GHz.
        max_delta : float
            Maximum percent change in frequency between adaptive passes
            required for convergence. Default is 1 (%).
        max_passes : int
            Maximum number of adaptive passes allowed. Default is 10.
        real : bool
            Whether convergence should be based on real frequency. Default is True.
        min_pass : int
            Minimum number of adaptive passes that must be performed. Default is 1.
        min_c_pass : int
            Minimum number of passes that must meet convergence criteria. Default is 2.

        Notes
        -----
        This method creates an eigenmode analysis setup using the HFSS API.

        The setup created uses:
        - The first setup name from `self.setup_names`
        - Basis order set to 2

        The resulting setup is added to the HFSS design through
        `self.hfss.create_setup()`.
        """
        self.hfss.create_setup(name=self.setup_names[0], NumModes=num_modes, MinimumFrequency=f"{start_f}GHz", MaxDeltaFreq=max_delta, MaximumPasses=max_passes,
                               ConvergeOnRealFreq=real, MinimumPasses=min_pass, MinimumConvergedPasses=min_c_pass, BasisOrder=2, )

    def create_hfss_geometries(self):
        """
        Generate the full HFSS model.

        Steps
        -----
        1. Export parameters
        2. Create silicon chip
        3. Add resonator
        4. Add junction and pads
        5. Add cavity post
        6. Add all coupling pins
        7. Create vacuum/tunnel geometry
        8. Create eigenmode solution setup

        Returns
        -------
        None
        """
        self.export_numeric_parameters()
        self.hfss.modeler.create_box([0, 0, 0], ["chip_t", "chip_l", "chip_w"], name="Chip", material="silicon")
        self.hfss.mesh.assign_length_mesh(assignment=["Chip"],  maximum_length="0.15mm", inside_selection=True, name="Chip_mesh")
        self.readout_resonator()
        self.junction_pads()        
        self.hfss.modeler.create_cylinder(orientation="Z", origin=["p_cav_tunnel_x", "p_cav_tunnel_y", "p_cav_tunnel_z"], radius="p_cav_post_r", height="p_cav_post_l", name="P_cavity_post", material="pec")
        self.readout_pin()
        self.qubit_pin()
        self.storage_pin()
        self.vacuum()
        self.solution_setup()

# ============================================================
# PyEPR analsyis with extension for SNAILs
# ============================================================

class pyepr_transmon:
    """
    Full PyEPR-based analysis pipeline for SNAIL/Transmon circuits.

    This class performs:
        - Distributed HFSS → PyEPR extraction
        - Quantum analysis
        - SNAIL nonlinear coefficient evaluation (c2, c3, c4)
        - Renormalization due to participation and shunt inductances
        - Kerr and cross-Kerr calculation
        - Three-wave and four-wave mixing corrections for SNAILs.
        - CSV data export

    Attributes
    ----------
    epra : epr.QuantumAnalysis
        Object storing the quantum solutions.
    eprh : epr.DistributedAnalysis
        Object storing the distributed HFSS extraction.
    distributed_file : str
        Path to the saved distributed analysis file.
    """

    def _mats(self, var):
        """
        Return the EPR base matrices for a given variation.

        Parameters
        ----------
        var : int or str
            Sweep index for the inductance parameter.

        Returns
        -------
        (Pmat, Cmat, Emat, Pvm, ZPF) : tuple of np.ndarray
            Full EPR matrices:
                Pmat : inductive participation matrix
                Cmat : capacitance matrix
                Emat : modal frequencies (GHz)
                Pvm  : normalization matrix
                ZPF  : zero-point fluctuations
        """
        #Shortcut for EPR base matrices.
        return self.epra.get_epr_base_matrices(variation=str(var), _renorm_pj=None)

    def _freq(self, mode, var):
        """
        Return angular frequency of a given mode (Hz).

        Parameters
        ----------
        mode : int
            Mode index.
        var : int
            Sweep variation index.

        Returns
        -------
        float
            Angular frequency in Hz.
        """
        mats = self._mats(var)
        return 2 * np.pi * mats[2][mode, mode] * 1e9

    def _compute_cparams(self, Phi_ext, a, n):
        """
        Compute SNAIL polynomial coefficients.

        Parameters
        ----------
        Phi_ext : float
            External flux (radians).
        a : float
            SNAIL asymmetry parameter.
        n : int
            Number of small junctions.

        Returns
        -------
        (c2, c3, c4) : tuple of floats
            Derivatives of the potential evaluated at the minimum.
        """
        c2 = self.c(Phi_ext, a, n, 2)
        c3 = self.c(Phi_ext, a, n, 3)
        c4 = self.c(Phi_ext, a, n, 4)
        return c2, c3, c4

    def _compute_L_params(self, Lj, c2, var):
        """
        Compute linearized SNAIL inductances.

        Parameters
        ----------
        Lj : float
            Josephson inductance in Henries.
        c2 : float
            Second derivative of potential.
        var : int
            Variation index.

        Returns
        -------
        (Ls, L0, p) : tuple of floats
            Ls : series inductance
            L0 : shunt inductance
            p  : participation ratio
        """
        Pmat = self._mats(var)[0]
        Ls = Lj / c2
        L0 = Ls * (1 - np.sum(Pmat[0]))
        p = Ls / (Ls + L0)
        return Ls, L0, p

    def _renormalize_coeffs(self, c2, c3, c4, p, M):
        """
        Renormalize nonlinear coefficients due to participation.

        Returns
        -------
        (c2t, c3t, c4t) : tuple of floats
            Renormalized c2, c3, c4.
        """
        c2t = p/M * c2
        c3t = p**3 / M**2 * c3
        c4t = p**4 / M**3 * (c4 - 3*c3**2/c2 * (1-p))
        return c2t, c3t, c4t

    # ============================================================
    # Distributed Analysis
    # ============================================================

    def distributed_analysis(self, filepath, my_project_name, my_design_name):
        pinfo = epr.ProjectInfo(
            project_path=filepath,
            project_name=my_project_name,
            design_name=my_design_name
        )
        """
        Run PyEPR distributed analysis on the HFSS project.

        Parameters
        ----------
        filepath : str
            Path to the HFSS project file.
        my_project_name : str
            Name of the project inside the file.
        my_design_name : str
            Name of the HFSS design.

        Side Effects
        ------------
        - Creates ``self.eprh`` (DistributedAnalysis object).
        - Saves distributed extraction file.
        - Releases HFSS session.
        """
        pinfo.junctions['j1'] = {'Lj_variable': 'Lj', 'rect': 'Junction', 'line': 'Junction_polyline'}
        pinfo.validate_junction_info()

        self.eprh = epr.DistributedAnalysis(pinfo)
        self.eprh.do_EPR_analysis([], modes=[0, 1, 2, 3], append_analysis=False)
        self.eprh.save()

        self.distributed_file = self.eprh.data_filename
        epr.ansys.release()

    # ============================================================
    # SNAIL nonlinear derivatives
    # ============================================================

    def c(self, Phi_ext, a, n, k):
        """
        Compute the k-th derivative of the SNAIL potential at its minimum.

        Parameters
        ----------
        Phi_ext : float
            External flux (radians).
        a : float
            Asymmetry parameter.
        n : int
            Number of small junctions.
        k : int
            Derivative order (2, 3, or 4).

        Returns
        -------
        float
            Value of d^k U / dPhi^k evaluated at the minimum.
        """
        def f(Phi_s):
            return abs(a*np.sin(Phi_s) + np.sin((Phi_s - Phi_ext)/n))

        res = minimize_scalar(f, method='bounded', bounds=(-m.pi, m.pi))
        phi_min = res.x

        Phi_s = Symbol('Phi_s')
        Us = -a * sym.cos(Phi_s) - n * sym.cos((Phi_ext - Phi_s)/n)
        d = Us.diff(Phi_s, k)

        return N(d.evalf(subs={Phi_s: phi_min}))

    # ============================================================
    # g-coefficient helper
    # ============================================================

    def g(self, ct, k, w, p0, c2t, var):
        """
        Compute nonlinear coupling coefficient g_k.

        Parameters
        ----------
        ct : float
            Renormalized nonlinear coefficient (c3t or c4t).
        k : int
            Nonlinearity order (3 or 4).
        w : float
            Mode frequency (Hz).
        p0 : float
            Participation for the target JJ mode.
        c2t : float
            Renormalized quadratic coefficient.
        var : int
            Variation index.

        Returns
        -------
        float
            Nonlinear coupling strength g_k (in Hz).
        """
        Phi_zpf = self._mats(var)[4][self.snail_mode, 0]
        return ct/(c2t * factorial(k)) * w/2 * (float(Phi_zpf))**(k-2)

    # ============================================================
    # 4-Wave Mixing
    # ============================================================

    def main_4wave_mixing(self, n, M, a, Phi_ratio, Lj, var):
        """
        Compute Kerr, cross-Kerr, and nonlinear couplings for 4-wave mixing.

        Returns
        -------
        tuple
            (K, χ_readout, χ_cavity, χ_box, g3, g4, Ls, L0)
        """
        mats = self._mats(var)
        Pmat, Emat, ZPF = mats[0], mats[2], mats[4]

        ps = Pmat[self.snail_mode]
        w = Emat[self.snail_mode, self.snail_mode] * 1e9

        # Kerr terms
        kxbbq = self.epra.get_chis(m=self.snail_mode, n=self.snail_mode, numeric=False)[int(var)]
        xsreadout = self.epra.get_chis(m=self.snail_mode, n=self.readout_mode, numeric=False)[int(var)]
        xscavity  = self.epra.get_chis(m=self.snail_mode, n=self.cavity_mode, numeric=False)[int(var)]
        xsbox     = self.epra.get_chis(m=self.snail_mode, n=self.box_mode, numeric=False)[int(var)]

        Phi_ext = 2*np.pi*Phi_ratio
        c2, c3, c4 = self._compute_cparams(Phi_ext, a, n)

        Ls, L0, p = self._compute_L_params(Lj, c2, var)
        c2t, c3t, c4t = self._renormalize_coeffs(c2, c3, c4, p, M)

        self.Ls, self.L0 = Ls, L0

        # nonlinear couplings
        self.g3_4 = self.g(c3t, 3, w, ps, c2t, var)
        self.g4_4 = self.g(c4t, 4, w, ps, c2t, var)

        # Kerr
        self.k_4 = c4t/c2t * kxbbq / 2
        self.crosskerrsreadout_4 = c4t/c2t * xsreadout
        self.crosskerrcavity_4  = c4t/c2t * xscavity
        self.crosskerrsbox_4    = c4t/c2t * xsbox

        return (
            self.k_4,
            self.crosskerrsreadout_4,
            self.crosskerrcavity_4,
            self.crosskerrsbox_4,
            self.g3_4,
            self.g4_4,
            self.Ls,
            self.L0
        )

    # ============================================================
    # Full cross-Kerr calculator
    # ============================================================

    def get_crosskerr_pq(self, p, q, g3, g4, var):
        """
        Compute cross-Kerr between modes p and q.

        Parameters
        ----------
        p, q : int
            Mode indices.
        g3, g4 : float
            3-wave and 4-wave nonlinear strengths.
        var : int
            Variation index.

        Returns
        -------
        float
            Cross-Kerr coefficient.
        """
        mats = self._mats(var)
        Emat = mats[2]
        Pmat = mats[0]

        omegas = 2*np.pi * np.diag(Emat) * 1e9

        omega_p = self._freq(p, var)
        omega_q = self._freq(q, var)

        omega_p = omega_p * np.ones_like(omegas)
        omega_q = omega_q * np.ones_like(omegas)

        omegaT_inv = (
            1/(omega_p + omega_q - omegas)
            + 1/(omega_p - omega_q - omegas)
            + 1/(-omega_p + omega_q - omegas)
            - 1/(omega_p + omega_q + omegas)
        )

        pp = Pmat[p]
        pq = Pmat[q]
        pvec = Pmat.T

        Kpq = pp * pq * (12*g4 + 18*g3**2 * (pvec @ omegaT_inv))
        return Kpq

    # ============================================================
    # 3-Wave Mixing
    # ============================================================

    def main_3wave_mixing(self, n, M, a, Phi_ratio, Lj, var):
        """
        Compute 3-wave mixing nonlinearities and Kerr terms.

        Returns
        -------
        tuple
            Kerr and cross-Kerr coefficients, and nonlinear couplings.
        """
        mats = self._mats(var)
        Pmat, Emat = mats[0], mats[2]

        ps = Pmat[self.snail_mode]
        w = Emat[self.snail_mode, self.snail_mode] * 1e9

        Phi_ext = 2*np.pi * Phi_ratio
        c2, c3, c4 = self._compute_cparams(Phi_ext, a, n)

        Ls, L0, p = self._compute_L_params(Lj, c2, var)
        c2t, c3t, c4t = self._renormalize_coeffs(c2, c3, c4, p, M)

        # nonlinear couplings
        self.g3_3 = self.g(c3t, 3, w, ps, c2t, var)
        self.g4_3 = self.g(c4t, 4, w, ps, c2t, var)

        # Kerr terms
        self.k_3 = self.get_crosskerr_pq(self.snail_mode, self.snail_mode, self.g3_3, self.g4_3, var)*1e-6
        self.crosskerrscavity_3     = self.get_crosskerr_pq(self.snail_mode, self.cavity_mode, self.g3_3, self.g4_3, var)*1e-6
        self.crosskerrsbox_3   = self.get_crosskerr_pq(self.snail_mode, self.box_mode, self.g3_3, self.g4_3, var)*1e-6
        self.crosskerrsreadout_3 = self.get_crosskerr_pq(self.snail_mode, self.readout_mode, self.g3_3, self.g4_3, var)*1e-6

        return (
            self.k_3,
            self.crosskerrscavity_3,
            self.crosskerrsbox_3,
            self.crosskerrsreadout_3,
            self.g3_3,
            self.g4_3
        )

    # ============================================================
    # Quantum Analysis
    # ============================================================

    def quantum_analysis(self, distributed_file, new_folder_path, a=0.219):
        """
        Run PyEPR quantum analysis on the distributed result.

        Parameters
        ----------
        distributed_file : str
            File generated by DistributedAnalysis.
        new_folder_path : str
            Path where results will be saved.
        a : float, optional
            SNAIL asymmetry parameter.

        Side Effects
        ------------
        - Creates `self.epra`
        - Computes frequencies, chis, participations.
        - Saves output folder.
        """
        
        self.epra = epr.QuantumAnalysis(distributed_file)
        self.epra.analyze_all_variations(cos_trunc=8, fock_trunc=10)

        sv = 'Lj'
        self.part_norm = pd.DataFrame(self.epra.get_participations(swp_variable=sv, inductive=True, _normed=True))
        self.part      = pd.DataFrame(self.epra.get_participations(swp_variable=sv, inductive=True, _normed=False))
        self.chis_nd   = self.epra.get_chis(swp_variable=sv, numeric=True)
        self.chis      = self.epra.get_chis(swp_variable=sv, numeric=False)
        self.frequencies_nd = self.epra.get_frequencies(swp_variable=sv, numeric=True)
        self.frequencies    = self.epra.get_frequencies(swp_variable=sv, numeric=False)

        os.makedirs(new_folder_path, exist_ok=True)

    # ============================================================
    # Saving CSVs
    # ============================================================
   
    def save_quantum_results(self, new_folder_path):
        """
        Store nonlinear coefficients and Kerrs into pandas DataFrames.

        Parameters
        ----------
        new_folder_path : str
            Destination folder.

        Returns
        -------
        snail_data_4 : pd.DataFrame
            4-wave mixing coefficients.
        snail_data_3 : pd.DataFrame
            3-wave mixing coefficients.
        """
        # 4‑wave
        #       
        snail_data_4=pd.concat([abs(pd.DataFrame([self.k_4])),abs(pd.DataFrame([self.crosskerrsreadout_4])),abs(pd.DataFrame([self.crosskerrsbox_4])),abs(pd.DataFrame([self.crosskerrcavity_4])),abs(pd.DataFrame([self.g4_4*1e-6])),abs(pd.DataFrame([self.g3_4*1e-6]))],axis=1)
        snail_data_4.columns=['K_4 (MHz)','crosskerrsreadout_4 (MHz)','crosskerrsbox_4 (MHz)','crosskerrcavity_4 (MHz)','g4_4 (MHz)','g3_4 (MHz)']

        # 3-wave
        #
        snail_data_3=pd.concat([abs(pd.DataFrame([self.k_3])),abs(pd.DataFrame([self.crosskerrsreadout_3])),abs(pd.DataFrame([self.crosskerrsbox_3])),abs(pd.DataFrame([self.crosskerrscavity_3])),abs(pd.DataFrame([self.g4_3*1e-6])),abs(pd.DataFrame([self.g3_3*1e-6]))],axis=1)
        snail_data_3.columns=['K_4 (MHz)','crosskerrreadout_3 (MHz)','crosskerrsbox_3 (MHz)','crosskerrcavity_3 (MHz)','g4_3 (MHz)','g3_3 (MHz)']

        snail_data_4.to_csv(str(new_folder_path)+"/snail_data_4_wave_mixing_correction.csv")
        snail_data_3.to_csv(str(new_folder_path)+"/snail_data_3_wave_mixing_correction.csv")
        
        return snail_data_4, snail_data_3
