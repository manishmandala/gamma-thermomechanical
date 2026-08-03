# Parameterized runner for *EXTERNAL_MESH control files - takes the control
# file and output folder as CLI args instead of hand-copying a new .py file
# per material (see legacy_main_wall_pure_TI64.py/pure_IN718.py/pure_Cu.py - flagged as tech debt;
# this is the fix for that pattern going forward). Generalizes
# main_external_mesh_test.py's logic (left untouched as a reference example)
# so pure-IN718, pure-1018, and graded runs on Rowan's mesh all share one
# script.
#
# Usage: python3 run_external_mesh.py <control_file> <output_dir>

import sys
import os

import cupy as cp
import numpy as np
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
cp.cuda.Device(0).use()
import pyvista as pv
import vtk
import meshio

CONTROL_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
SOURCE_MESH = 'sol_100.vtu'   # provides the initial T field - same for every run, so material is the only variable
STOP_TIME = 5.0
N_FRAMES = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

# sort_birth=False: keeps node/element order identical to sol_100.vtu's own
# row order across all runs (everything here has birth=0, so this isn't about
# birth sequencing - it's what lets a comparison script pick probe nodes/
# elements by a fixed index and have that index mean the same physical
# location in every run).
domain = domain_mgr(filename=CONTROL_FILE, sort_birth=False)
heat_solver = heat_solve_mgr(domain)

ext = meshio.read(SOURCE_MESH)
heat_solver.temperature[:] = cp.asarray(ext.point_data['T'].astype(np.float64).ravel())

n_n = len(domain.nodes)
n_e = len(domain.elements)


def save_vtk(filename, time_value):
    n_e_save = int(cp.sum(domain.active_elements))
    n_n_save = int(cp.sum(domain.active_nodes))
    active_elements = domain.elements[domain.active_elements].tolist()
    active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
    active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
    points = domain.nodes[0:n_n_save].get()
    active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
    active_grid.cell_data['material'] = domain.element_mat[domain.active_elements]
    if len(domain.mat_graded) > 0:
        active_grid.cell_data['composition'] = domain.element_composition[domain.active_elements].get()
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.field_data['TIME'] = np.array([time_value])
    active_grid.save(filename)


file_num = 0
save_vtk(os.path.join(OUTPUT_DIR, 'wall_{:04d}.vtu'.format(file_num)), 0.0)
file_num += 1

next_save_time = STOP_TIME / N_FRAMES
save_interval = next_save_time
t = 0
bar_len = 30

while domain.current_sim_time < STOP_TIME:
    t += 1
    heat_solver.time_integration()

    if domain.current_sim_time >= next_save_time:
        save_vtk(os.path.join(OUTPUT_DIR, 'wall_{:04d}.vtu'.format(file_num)), float(domain.current_sim_time))
        file_num += 1
        next_save_time += save_interval

    if t % 200 == 0:
        frac = min(domain.current_sim_time / STOP_TIME, 1.0)
        filled = int(bar_len * frac)
        bar = '#' * filled + '-' * (bar_len - filled)
        print('\r|{}| {:.1f}%   sim time {:.2f}s / {:.2f}s'.format(bar, 100*frac, domain.current_sim_time, STOP_TIME), end='', flush=True)

    if t % 5000 == 0:
        cp.get_default_memory_pool().free_all_blocks()

save_vtk(os.path.join(OUTPUT_DIR, 'wall_{:04d}.vtu'.format(file_num)), float(domain.current_sim_time))
temp = heat_solver.temperature.get()
print('\nDone - {} frames written to {}/'.format(file_num + 1, OUTPUT_DIR))
print('final temp range: {:.1f} .. {:.1f}'.format(temp.min(), temp.max()))
