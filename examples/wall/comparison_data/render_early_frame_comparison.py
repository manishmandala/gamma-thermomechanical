import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pv.OFF_SCREEN = True

# Frame chosen where the Cu-vs-graded contrast is near its peak within the
# transient window (before Cu's fast conduction washes it back out) - see
# the per-frame maxT scan: Cu ~1406K vs Graded ~1699K at this frame.
FRAME = 'wall_0010.vtu'
CLIM = [300, 1750]

runs = [
    ('Pure Cu', f'../results_Cu/{FRAME}'),
    ('Graded TI64/Cu', f'../results_graded_cu/{FRAME}'),
]

images = []
for label, path in runs:
    mesh = pv.read(path)
    t = float(mesh.field_data['TIME'][0])
    p = pv.Plotter(off_screen=True, window_size=[900, 500])
    p.add_mesh(mesh, scalars='temp', clim=CLIM, cmap='inferno', show_scalar_bar=(label == runs[-1][0]))
    p.camera_position = [(20, -25, 15), (0, 0, 0.5), (0, 0, 1)]
    p.camera.zoom(2.2)
    img_path = f'/tmp/render_{label.replace(" ", "_").replace("/", "-")}.png'
    p.screenshot(img_path)
    images.append((label, t, img_path))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (label, t, img_path) in zip(axes, images):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(f'{label}  (t={t:.3f}s)')
    ax.axis('off')

fig.suptitle('Temperature field, same frame/timestep, fixed color scale 300-1750K', fontsize=12)
fig.tight_layout()
out_path = 'early_frame_temp_comparison.png'
fig.savefig(out_path, dpi=180)
print(f'Saved {out_path}')
