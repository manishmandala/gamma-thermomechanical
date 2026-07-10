import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cusparse
import cupyx.scipy.sparse.linalg
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
from gamma.simulator.func import elastic_stiff_matrix,constitutive_problem,transformation,disp_match
cp.cuda.Device(0).use()
import pyvista as pv
import vtk


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
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.point_data['U1'] = U[0:n_n_save,0].get()
    active_grid.point_data['U2'] = U[0:n_n_save,1].get()
    active_grid.point_data['U3'] = U[0:n_n_save,2].get()
    active_grid.field_data['TIME'] = np.array([time_value])   # so ParaView's time slider shows real seconds, not file index
    active_grid.save(filename)


domain = domain_mgr(filename='thinwall.k')
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
filename = 'results/wall_{:04d}.vtk'.format(file_num)
save_vtk(filename, 0.0)
file_num = file_num + 1

# ---- run settings ----
STOP_FRACTION = 0.3     # 0.10 = 10% preview. Set to 1.0 for the real full run.
N_FRAMES = 200           # more frames so the fast (1s) deposit sweeps get several frames each, not just the long dwell
stop_time = STOP_FRACTION * endtime
save_interval = stop_time / N_FRAMES
next_save_time = save_interval
bar_len = 30

# ---- temperature logging at the base-substrate node ----
node_id = 3710           # base-center node (x=0, y=-0.1, z=-4.0)
log_every = 20
time_hist = []
temp_hist = []

while domain.current_sim_time < endtime - domain.dt:
    t = t + 1
    heat_solver.time_integration()

    if t % log_every == 0:
        time_hist.append(float(domain.current_sim_time))
        temp_hist.append(float(heat_solver.temperature[node_id]))

    if domain.current_sim_time >= next_save_time:
        save_vtk('results/wall_{:04d}.vtk'.format(file_num), float(domain.current_sim_time))
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
        save_vtk('results/wall_{:04d}.vtk'.format(file_num), float(domain.current_sim_time))
        print('\nReached {:.0f}% of the build - stopping early.'.format(100*STOP_FRACTION))
        break

# ---- write the temperature history to a CSV after the run ----
import csv
with open('temperature_history.csv', 'w', newline='') as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(['time_s', 'temperature_K'])
    for tt, TT in zip(time_hist, temp_hist):
        writer.writerow([tt, TT])
print('\nSaved temperature_history.csv with {} rows'.format(len(time_hist)))
