# Proves out *EXTERNAL_MESH import against Rowan's sol_100.vtu: loads the
# geometry only (no toolpath, no composition import yet - see roadmap notes),
# seeds the initial temperature from the file's own T point-data field
# (200-800K), and lets it diffuse under conduction + boundary convection/
# radiation with the laser held off (toolpath_off.crs). Placeholder TI64
# properties everywhere - not physically meaningful for IN625/SS316/HA25/
# MCrAlY, this is purely a mesh-import + solver-plumbing sanity check.

import cupy as cp
import numpy as np
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
cp.cuda.Device(0).use()
import pyvista as pv
import vtk
import meshio
import os

os.makedirs('results_external_mesh', exist_ok=True)

domain = domain_mgr(filename='sol_100_control.k', sort_birth=False)
heat_solver = heat_solve_mgr(domain)

# seed initial temperature from the source file's own T field (node order is
# preserved 1:1 since sort_birth=False)
ext = meshio.read('sol_100.vtu')
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
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.field_data['TIME'] = np.array([time_value])
    active_grid.save(filename)


file_num = 0
save_vtk('results_external_mesh/wall_{:04d}.vtk'.format(file_num), 0.0)
file_num += 1

STOP_TIME = 5.0
N_FRAMES = 40
save_interval = STOP_TIME / N_FRAMES
next_save_time = save_interval
t = 0
bar_len = 30

while domain.current_sim_time < STOP_TIME:
    t += 1
    heat_solver.time_integration()

    if domain.current_sim_time >= next_save_time:
        save_vtk('results_external_mesh/wall_{:04d}.vtk'.format(file_num), float(domain.current_sim_time))
        file_num += 1
        next_save_time += save_interval

    if t % 200 == 0:
        frac = min(domain.current_sim_time / STOP_TIME, 1.0)
        filled = int(bar_len * frac)
        bar = '#' * filled + '-' * (bar_len - filled)
        print('\r|{}| {:.1f}%   sim time {:.2f}s / {:.2f}s'.format(bar, 100*frac, domain.current_sim_time, STOP_TIME), end='', flush=True)

    if t % 5000 == 0:
        cp.get_default_memory_pool().free_all_blocks()

save_vtk('results_external_mesh/wall_{:04d}.vtk'.format(file_num), float(domain.current_sim_time))
temp = heat_solver.temperature.get()
print('\nDone - {} frames written to results_external_mesh/'.format(file_num + 1))
print('final temp range: {:.1f} .. {:.1f} (started 200.0 .. 800.0)'.format(temp.min(), temp.max()))
