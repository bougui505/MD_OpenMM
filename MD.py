#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-02-03 12:33:21 (UTC+0100)

from __future__ import print_function

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout
import sys
import itertools

class MD:
    """
    """
    def __init__(self, filename_pdb, forcefield_name='amber99sb.xml',
                 water_model_name='tip3p.xml'):
        self.filename_pdb = filename_pdb
        self.pdb = app.PDBFile(filename_pdb)
        self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = app.ForceField(forcefield_name, water_model_name)
        self.system = None
        self.integrator = None
        self.platform = None
        self.simulation = None

    def add_solvent(self, padding=1.0*unit.nanometers,
                    ionicStrength=0.1*unit.molar):
        print("Adding solvent...")
        self.modeller.addSolvent(self.forcefield, padding=padding,
                                 ionicStrength=ionicStrength)
        print("done")

    def create_system(self, nonbondedMethod=app.PME,
                    nonbondedCutoff=1.0*unit.nanometers,
                    constraints=app.HBonds, rigidWater=True,
                    ewaldErrorTolerance=0.0005, temperature=300*unit.kelvin,
                    collision_rate=1.0/unit.picoseconds,
                    timestep=2.0*unit.femtoseconds, platform_name='OpenCL',
                    CpuThreads=1, external_force=None,
                    global_parameters=[('k', 5.0*unit.kilocalories_per_mole\
                    /unit.angstroms**2)],
                    per_particle_parameters=['x0', 'y0', 'z0'],
                    atom_list=('CA', 'C', 'N', 'O'),
                    per_particle_values=[]):
        """
        • If external_force is not None: add the algebraic expression as an
        external force to the system (see: https://goo.gl/ei4pza)
        ‣ Positional restraints example:
            external_force = "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        ‣ The global_parameters gives the global parameters for the external force 
        and the default is set for the positional restraints example.
        ‣ The per_particle_parameters gives the per particle parameters and the
        default is set to the x, y, z coordinates. The 3 first
        per_particle_parameters must always be the 'x0', 'y0', 'z0' coordinates.
        ‣ atom_list is the list of atom type to apply the force on
        ‣ The per_particle_values gives the values for the fourth per particle
        parameters if it exists. This parameter is assumed to be a distance in
        Angstrom.
        """
        print("Creating system...")
        per_particle_values = itertools.chain(per_particle_values)
        self.system = self.forcefield.createSystem(self.modeller.topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
            constraints=constraints, rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance)
        self.integrator = mm.LangevinIntegrator(temperature, collision_rate,
                                                timestep)
        self.platform = mm.Platform.getPlatformByName(platform_name)
        if platform_name == 'CPU':
            self.platform.setPropertyDefaultValue('CpuThreads', '%s'%CpuThreads)
        if external_force is not None:
            self.force = mm.CustomExternalForce(external_force)
            atom_iter = self.modeller.topology.atoms()
            for p,v in global_parameters:
                self.force.addGlobalParameter(p, v)
            for p in per_particle_parameters:
                self.force.addPerParticleParameter(p)
            for i, atom_crd in enumerate(self.modeller.positions):
                atom = atom_iter.next()
                if atom.name in atom_list and atom.residue.name != 'HOH':
                    if len(per_particle_parameters) == 3:
                        self.force.addParticle(i,
                                     atom_crd.value_in_unit(mm.unit.nanometers))
                    else: # More than the 3 coordinates as per particle parameters
                        #print(atom.name, atom.residue.name)
                        try:
                            values = list(atom_crd)
                            values.append(per_particle_values.next()*unit.angstroms)
                            values = unit.Quantity(values)
                            self.force.addParticle(i, values)
                        except StopIteration:
                            print("WARNING: The per_particle_values reach the end")
                            pass
            self.system.addForce(self.force)
            print("NumPerParticleParameters:%d"%self.force.getNumPerParticleParameters())
            print("NumParticles: %d"%self.force.getNumParticles())
            print("ParticleParameters 0: %s"%self.force.getParticleParameters(0))
        self.simulation = app.Simulation(self.modeller.topology, self.system,
                                        self.integrator, self.platform)
        self.simulation.context.setPositions(self.modeller.positions)
        self.simulation.context.setVelocitiesToTemperature(temperature)
        print("done")

    def minimize(self, filename_output_pdb="minimized.pdb",):
        print("Minimizing...")
        self.simulation.minimizeEnergy()
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(self.simulation.topology, positions,
                              open(filename_output_pdb, 'w'))
        print("done")

    def run(self, number_of_steps=15000, report_interval=500,
                    filename_output_dcd="traj.dcd",
                    filename_output_log="openmm_equilibration.log"):
        print("Running MD...")
        self.simulation.reporters.append(app.DCDReporter(filename_output_dcd,
                                                         report_interval))
        self.simulation.reporters.append(app.StateDataReporter(filename_output_log,
            report_interval, step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True, temperature=True, volume=True,
            density=True, progress=True, remainingTime=True, speed=True,
            totalSteps=number_of_steps, separator='\t'))
        self.simulation.step(number_of_steps)
        print("done")

    def forcegroupify(self):
        """
        Create force groups for the system
        (see https://goo.gl/p12IkM)
        """
        forcegroups = {}
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            forcegroups[force] = i
        return forcegroups

    def getEnergyDecomposition(self):
        """
        Decompose the energy for each force.
        Useful to see the energy of the CustomExternalForce
        (see https://goo.gl/p12IkM)
        """
        context = self.simulation.context
        forcegroups = self.forcegroupify()
        energies = {}
        for f, i in forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i)\
                            .getPotentialEnergy()
        return energies
