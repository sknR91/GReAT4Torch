import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from . import utils

class plot:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.plotted_image = None

    @staticmethod
    def plot_matrix_values(M, fignum=10101, title='', cmap='inferno', normalize=True):
        plt.figure(fignum)
        plt.clf()
        if normalize:
            plt.imshow((M) / np.max(M.flatten()), cmap=cmap)
        else:
            plt.imshow(M, cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

    def plot_grid_2d(self, phi, skip=None, newFig=True, fignr=101, **kwargs):
        """ Plot the grid specified by `phi` to the current plot/figure

        Args:
            phi : numpy array of shape (nx,ny,2)
            skip : (optional) skip every
            kwargs : further keyword arguments are forwarded to matplotlib
        """
        if not newFig:
            self.fig = plt.figure(fignr)

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if skip is None:
            m = phi.shape[0:-2]
            skip = np.round(np.ceil(np.log2(np.min(m)))/2).astype('int')

        for i in range(0, phi.shape[0], skip):
            self.plotted_image = plt.plot(phi[i, :, 0], phi[i, :, 1], **kwargs)
        for i in range(0, phi.shape[1], skip):
            self.plotted_image = plt.plot(phi[:, i, 0], phi[:, i, 1], **kwargs)

        # Make sure the upper and right boundaries are plotted
        if (phi.shape[0] - 1) % skip != 0:
            self.plotted_image = plt.plot(phi[-1, :, 0], phi[-1, :, 1], **kwargs)
            self.plotted_image = plt.plot(phi[:, -1, 0], phi[:, -1, 1], **kwargs)

    def plot_grid_3d(self, phi, skip=None, newFig=True, fignr=101, **kwargs):
        """ Plot the grid specified by `phi` to the current plot/figure

        Args:
            phi : numpy array of shape (nx,ny,nz,3)
            skip : (optional) skip every
            kwargs : further keyword arguments are forwarded to matplotlib
        """
        self.fig = plt.figure(fignr)
        self.fig.add_subplot(111, projection = '3d')
        if newFig:
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if skip is None:
            m = phi.shape[0:-1]
            skip = 2*int(0.75*np.round(np.ceil(np.log2(np.min(m)))).astype('int'))

        for j in range(0, phi.shape[2], skip):
            for i in range(0, phi.shape[0], skip):
                self.plotted_image = plt.plot(xs=phi[i, :, j, 0].ravel('F'), ys=phi[i, :, j, 1].ravel('F'),
                                              zs=phi[i, :, j, 2].ravel('F'), **kwargs)
            for i in range(0, phi.shape[1], skip):
                self.plotted_image = plt.plot(xs=phi[:, i, j, 0].ravel('F'), ys=phi[:, i, j, 1].ravel('F'),
                                              zs=phi[:, i, j, 2].ravel('F'), **kwargs)

        for i in range(0, phi.shape[0], skip):
            for j in range(0, phi.shape[1], skip):
                self.plotted_image = plt.plot(xs=phi[i, j, :, 0].ravel('F'), ys=phi[i, j, :, 1].ravel('F'),
                                              zs=phi[i, j, :, 2].ravel('F'), **kwargs)

        # Make sure the upper, right and rear boundaries are plotted
        if (phi.shape[0] - 1) % skip != 0 and (phi.shape[1] - 1) % skip != 0 and (phi.shape[2] - 1) % skip != 0:
            for j in range(0, phi.shape[2], skip):
                self.plotted_image = plt.plot(xs=phi[-1, :, j, 0].ravel('F'), ys=phi[-1, :, j, 1].ravel('F'),
                                              zs=phi[-1, :, j, 2].ravel('F'), **kwargs)
                self.plotted_image = plt.plot(xs=phi[:, -1, j, 0].ravel('F'), ys=phi[:, -1, j, 1].ravel('F'),
                                              zs=phi[:, -1, j, 2].ravel('F'), **kwargs)
            for j in range(0, phi.shape[0], skip):
                self.plotted_image = plt.plot(xs=phi[j, :, -1, 0].ravel('F'), ys=phi[j, :, -1, 1].ravel('F'),
                                              zs=phi[j, :, -1, 2].ravel('F'), **kwargs)
                self.plotted_image = plt.plot(xs=phi[j, -1, :, 0].ravel('F'), ys=phi[j, -1, :, 1].ravel('F'),
                                              zs=phi[j, -1, :, 2].ravel('F'), **kwargs)
            for j in range(0, phi.shape[1], skip):
                self.plotted_image = plt.plot(xs=phi[:, j, -1, 0].ravel('F'), ys=phi[:, j, -1, 1].ravel('F'),
                                              zs=phi[:, j, -1, 2].ravel('F'), **kwargs)
                self.plotted_image = plt.plot(xs=phi[-1, j, :, 0].ravel('F'), ys=phi[-1, j, :, 1].ravel('F'),
                                              zs=phi[-1, j, :, 2].ravel('F'), **kwargs)

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def scroll_image_2d(self, volume, omega=None, permute=None, colormap='gray', colorbar=False):
        if permute is not None:
            volume = np.transpose(volume, permute)

        self._remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        self.fig, self.ax = plt.subplots()
        self.ax.vol = volume
        self.ax.idx = volume.shape[-1] // 2
        self.ax.viewer = 'scrollView3'
        self.plotted_image = self.ax.imshow(volume[:, :, self.ax.idx], cmap=colormap, extent=omega,
                            norm=mpl.colors.Normalize(vmin=np.min(volume), vmax=np.max(volume)))
        self.fig.colorbar(self.plotted_image) if colorbar else 1
        ttl = 'Position: {0} of {1}'.format(self.ax.idx,volume.shape[-1])
        self.ax.set_xlabel(ttl)
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)

    def scroll_image_3d(self, volume, omega=None, permute=None):
        if permute is not None:
            volume = np.transpose(volume, permute)
            if omega is not None:
                if len(omega) < len(volume.shape):
                    print('Warning: Omega should be provided in full length!')
                omega = omega[np.array([1,3,5,7])]
                omega = omega[np.array(permute)]
                omega = omega[0:2]
                omega = utils.get_omega(omega)

        self._remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        self.fig, self.ax = plt.subplots()
        self.ax.vol = volume
        self.ax.idx = volume.shape[-2] // 2
        self.ax.tpnt = volume.shape[-1] // 2
        self.ax.viewer = 'scrollView4'
        self.ax.imshow(volume[:, :, self.ax.idx, self.ax.tpnt], cmap='gray', extent=omega)
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)

    def scroll_grid_2d(self, gridStack):
        self._remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        self.fig, self.ax = plt.subplots()
        self.ax.vol = gridStack
        self.ax.idx = gridStack.shape[-1] // 2
        self.ax.viewer = 'scrollGrid2'
        self.plot_grid_2d(gridStack[:, :, :, self.ax.idx])
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)

    def scroll_grid_3d(self, gridStack, skip=None, newFig=True):
        self._remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        self.fig, self.ax = plt.subplots()
        self.ax.vol = gridStack
        self.ax.idx = gridStack.shape[-1] // 2
        self.ax.viewer = 'scrollGrid3'
        self.plot_grid_3d(gridStack[:, :, :, :, self.ax.idx], skip=skip, newFig=newFig)
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)

    def _process_key(self, event):
        self.fig = event.canvas.figure
        self.ax = self.fig.axes[0]
        if event.key == 'w' or event.key == 'up': # arrow up
            self._previous_slice()
        elif event.key == 's' or event.key == 'down': # arrow down
            self._next_slice()
        elif (event.key == 'a' or event.key == 'left') and self.ax.viewer == 'scrollView4': # arrow left
            self._previous_timepoint()
        elif (event.key == 'd' or event.key == 'right') and self.ax.viewer == 'scrollView4': # arrow right
            self._next_timepoint()
        self.fig.canvas.draw()

    def _previous_slice(self):
        vol = self.ax.vol
        if self.ax.viewer == 'scrollView3':
            self.ax.idx = (self.ax.idx - 1) % vol.shape[-1]
            self.plotted_image.set_array(vol[:, :, self.ax.idx])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollView4':
            self.ax.idx = (self.ax.idx - 1) % vol.shape[-2]
            self.plotted_image.set_array(vol[:,:,self.ax.idx,self.ax.tpnt])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollGrid2':
            self.ax.idx = (self.ax.idx - 1) % vol.shape[-1]
            plt.gcf(); plt.cla()
            self.plot_grid_2d(vol[:, :, :, self.ax.idx])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollGrid3':
            self.ax.idx = (self.ax.idx - 1) % vol.shape[-1]
            plt.gca(); plt.cla()
            self.plot_grid_3d(vol[:, :, :, :, self.ax.idx], newFig=False)
            self.fig.canvas.draw_idle()
        ttl = 'Position: {0} of {1}'.format(self.ax.idx, vol.shape[-1])
        self.ax.set_xlabel(ttl)

    def _previous_timepoint(self):
        vol = self.ax.vol
        self.ax.tpnt = (self.ax.tpnt - 1) % vol.shape[-1]
        self.plotted_image.set_array(vol[:,:,self.ax.idx,self.ax.tpnt])

    def _next_slice(self):
        vol = self.ax.vol
        if self.ax.viewer == 'scrollView3':
            self.ax.idx = (self.ax.idx + 1) % vol.shape[-1]
            self.plotted_image.set_array(vol[:, :, self.ax.idx])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollView4':
            self.ax.idx = (self.ax.idx + 1) % vol.shape[-2]
            self.plotted_image.set_array(vol[:, :, self.ax.idx, self.ax.tpnt])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollGrid2':
            self.ax.idx = (self.ax.idx + 1) % vol.shape[-1]
            plt.gcf(); plt.cla()
            self.plot_grid_2d(vol[:, :, :, self.ax.idx])
            self.fig.canvas.draw_idle()
        elif self.ax.viewer == 'scrollGrid3':
            self.ax.idx = (self.ax.idx + 1) % vol.shape[-1]
            plt.gca(); plt.cla()
            self.plot_grid_3d(vol[:, :, :, :, self.ax.idx], newFig=False)
            self.fig.canvas.draw_idle()
        ttl = 'Position: {0} of {1}'.format(self.ax.idx, vol.shape[-1])
        self.ax.set_xlabel(ttl)

    def _next_timepoint(self):
        vol = self.ax.vol
        self.ax.tpnt = (self.ax.tpnt + 1) % vol.shape[-1]
        self.plotted_image.set_array(vol[:, :, self.ax.idx, self.ax.tpnt])

    def plot_mip(self, volume, omega=None, permute=None, colormap='gray', colorbar=False, fliptranspose=False):
        if permute is not None:
            volume = np.transpose(volume, permute)

        if fliptranspose:
            volume = np.flipud(volume.transpose((1, 0, 2)))

        self.fig, self.ax = plt.subplots()
        mip = np.max(volume, axis=2)
        self.plotted_image = self.ax.imshow(mip, cmap=colormap, extent=omega)#,norm=mpl.colors.Normalize(vmin=np.min(volume),vmax=np.max(volume)))
        self.fig.colorbar(self.plotted_image) if colorbar else 1