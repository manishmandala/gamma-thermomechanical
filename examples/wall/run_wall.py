# Unified thermal(-mechanical) simulation runner for any wall-example .k
# file - point it at a deck and an output folder via CLI args instead of
# hand-editing a script per material/composition mode. This is the ONLY
# runner used throughout the toolpath-arc-length composition project
# (examples/incoming_dataset/part00N_.../) and the only way to start a
# simulation anywhere in examples/wall/.

import argparse
import os

parser = argparse.ArgumentParser(
    description='Run a thin-wall DED thermal(-mechanical) sim from a .k file. '
                 'Replaces the old per-material main_wall_<material>.py copies - '
                 'point it at any .k file and an output folder instead of hand-editing a script.')
parser.add_argument('--k', required=True, help='input .k file, e.g. thinwall_IN718.k')
parser.add_argument('--out', required=True, help='output folder for result frames, e.g. results/pure_IN718')
parser.add_argument('--toolpath', default='toolpath.crs',
                     help='toolpath .crs file, resolved against --input-data-dir (not the cwd) - the '
                          '*TOOL_FILE line inside the .k is ignored by this solver, so this is the only '
                          'way to point at one. FIXED 2026-07-31: previously resolved against the cwd '
                          'regardless of --input-data-dir, which silently loaded examples/wall/toolpath.crs '
                          '(the wall demo\'s own short toolpath) whenever a bare filename was passed '
                          'alongside a non-default --input-data-dir - see memory/project notes for the '
                          'affected historical runs this produced.')
parser.add_argument('--input-data-dir', default='.',
                     help='base dir that the .k file\'s relative curve/material paths (e.g. ./materials/foo_cp.txt) '
                          'are resolved against (default: cwd)')
parser.add_argument('--stop-fraction', type=float, default=0.5,
                     help='fraction of the toolpath end time to simulate (default 0.5)')
parser.add_argument('--n-frames', type=int, default=200, help='number of saved frames (default 200)')
parser.add_argument('--device', type=int, default=0, help='CUDA device id (default 0)')
parser.add_argument('--format', choices=['vtu', 'vtk'], default='vtu',
                     help='output format - vtu is preferred, legacy vtk silently drops every '
                          'cell array after the first (default vtu)')
parser.add_argument('--log-node', type=int, default=None,
                     help='optional node id to log a temperature_history.csv for (default: no logging)')
parser.add_argument('--log-every', type=int, default=20, help='steps between log-node samples (default 20)')
args = parser.parse_args()

import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cusparse
import cupyx.scipy.sparse.linalg
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
from gamma.simulator.func import elastic_stiff_matrix, constitutive_problem, transformation, disp_match
cp.cuda.Device(args.device).use()
import pyvista as pv
import vtk

os.makedirs(args.out, exist_ok=True)


def save_vtk(filename, time_value):
    n_e_save = cp.sum(domain.active_elements)
    n_n_save = cp.sum(domain.active_nodes)
    active_elements = domain.elements[domain.active_elements].tolist()
    active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
    active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
    points = domain.nodes[0:n_n_save].get() + 5*U[0:n_n_save].get()
    Sv =  transformation(cp.sqrt(1/2*((S[0:n_e_save,:,0]-S[0:n_e_save,:,1])**2 + (S[0:n_e_save,:,1]-S[0:n_e_save,:,2])**2 + (S[0:n_e_save,:,2]-S[0:n_e_save,:,0])**2 + 6*(S[0:n_e_save,:,3]**2+S[0:n_e_save,:,4]**2+S[0:n_e_save,:,5]**2))),domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S11 = transformation(S[0:n_e_save,:,0], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S22 = transformation(S[0:n_e_save,:,1], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S33 = transformation(S[0:n_e_save,:,2], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S12 = transformation(S[0:n_e_save,:,3], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S23 = transformation(S[0:n_e_save,:,4], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    S13 = transformation(S[0:n_e_save,:,5], domain.elements[0:n_e_save], ele_detJac[0:n_e_save],n_n_save)
    active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
    active_grid.cell_data['material'] = domain.element_mat[domain.active_elements]
    if domain.mat_graded:
        # material is flat across a graded region - composition is the field
        # that actually shows the gradient.
        active_grid.cell_data['composition'] = domain.element_composition[domain.active_elements].get()
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.point_data['U1'] = U[0:n_n_save,0].get()
    active_grid.point_data['U2'] = U[0:n_n_save,1].get()
    active_grid.point_data['U3'] = U[0:n_n_save,2].get()
    active_grid.field_data['TIME'] = np.array([time_value])   # so ParaView's time slider shows real seconds, not file index
    active_grid.save(filename)


domain = domain_mgr(filename=args.k, toolpathdir=os.path.join(args.input_data_dir, args.toolpath),
                     input_data_dir=args.input_data_dir)
heat_solver = heat_solve_mgr(domain)
endtime = domain.end_sim_time
n_n = len(domain.nodes)
n_e = len(domain.elements)
n_q = 8
file_num = 0

U = cp.zeros((n_n,3))
S = cp.zeros((n_e,n_q,6))
idirich = cp.array(domain.nodes[:, 2] == -4.0)

nodes_pos = domain.nodes[domain.elements]
Jac = cp.matmul(domain.Bip_ele,nodes_pos[:,cp.newaxis,:,:].repeat(8,axis=1))
ele_detJac = cp.linalg.det(Jac)

t = 0
filename = os.path.join(args.out, 'wall_{:04d}.{}'.format(file_num, args.format))
save_vtk(filename, 0.0)
file_num = file_num + 1

# ---- run settings ----
stop_time = args.stop_fraction * endtime
save_interval = stop_time / args.n_frames
next_save_time = save_interval
bar_len = 30

# ---- optional temperature logging at a chosen node ----
time_hist = []
temp_hist = []

while domain.current_sim_time < endtime - domain.dt:
    t = t + 1
    heat_solver.time_integration()

    if args.log_node is not None and t % args.log_every == 0:
        time_hist.append(float(domain.current_sim_time))
        temp_hist.append(float(heat_solver.temperature[args.log_node]))

    if domain.current_sim_time >= next_save_time:
        save_vtk(os.path.join(args.out, 'wall_{:04d}.{}'.format(file_num, args.format)), float(domain.current_sim_time))
        file_num = file_num + 1
        next_save_time = next_save_time + save_interval

    if t % 200 == 0:
        frac = min(domain.current_sim_time / stop_time, 1.0)
        filled = int(bar_len * frac)
        bar = '#' * filled + '-' * (bar_len - filled)
        print('\r|{}| {:.1f}%   sim time {:.2f}s / {:.2f}s'.format(bar, 100*frac, domain.current_sim_time, stop_time), end='', flush=True)

    if t % 5000 == 0:
        cp.get_default_memory_pool().free_all_blocks()

    if domain.current_sim_time >= stop_time:
        save_vtk(os.path.join(args.out, 'wall_{:04d}.{}'.format(file_num, args.format)), float(domain.current_sim_time))
        print('\nReached {:.0f}% of the build - stopping early.'.format(100*args.stop_fraction))
        break

print('\nDone - {} frames written to {}/'.format(file_num, args.out))

if args.log_node is not None:
    import csv
    with open('temperature_history.csv', 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['time_s', 'temperature_K'])
        for tt, TT in zip(time_hist, temp_hist):
            writer.writerow([tt, TT])
    print('Saved temperature_history.csv with {} rows'.format(len(time_hist)))
