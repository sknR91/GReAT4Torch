import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from . import utils

class plot:
    @staticmethod
    def plotMatrixValues(M, fignum=10101, title='', cmap='inferno', normalize=True):
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

    @staticmethod
    def plotGrid(phi, skip=None, newFig = True, fignr=101, **kwargs):
        """ Plot the grid specified by `phi` to the current plot/figure

        Args:
            phi : numpy array of shape (nx,ny,2)
            skip : (optional) skip every
            kwargs : further keyword arguments are forwarded to matplotlib
        """
        if not newFig:
            fig = plt.figure(fignr)

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if skip is None:
            m = phi.shape[0:-2]
            skip = np.round(np.ceil(np.log2(np.min(m)))/2).astype('int')

        for i in range(0, phi.shape[0], skip):
            plt.plot(phi[i, :, 0], phi[i, :, 1], **kwargs)
        for i in range(0, phi.shape[1], skip):
            plt.plot(phi[:, i, 0], phi[:, i, 1], **kwargs)

        # Make sure the upper and right boundaries are plotted
        if (phi.shape[0] - 1) % skip != 0:
            plt.plot(phi[-1, :, 0], phi[-1, :, 1], **kwargs)
            plt.plot(phi[:, -1, 0], phi[:, -1, 1], **kwargs)

    @staticmethod
    def plotGrid3(phi, skip=None, newFig = True, fignr=101, **kwargs):
        """ Plot the grid specified by `phi` to the current plot/figure

        Args:
            phi : numpy array of shape (nx,ny,nz,3)
            skip : (optional) skip every
            kwargs : further keyword arguments are forwarded to matplotlib
        """
        fig = plt.figure(fignr)
        fig.add_subplot(111, projection = '3d')
        if newFig:
            plt.subplots(subplot_kw={'projection': '3d'})

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if skip is None:
            m = phi.shape[0:-1]
            skip = 2*int(0.75*np.round(np.ceil(np.log2(np.min(m)))).astype('int'))

        for j in range(0, phi.shape[2], skip):
            for i in range(0, phi.shape[0], skip):
                plt.plot(xs=phi[i, :, j, 0].ravel('F'), ys=phi[i, :, j, 1].ravel('F'),  zs=phi[i, :, j, 2].ravel('F'), **kwargs)
            for i in range(0, phi.shape[1], skip):
                plt.plot(xs=phi[:, i, j, 0].ravel('F'), ys=phi[:, i, j, 1].ravel('F'), zs=phi[:, i, j, 2].ravel('F'), **kwargs)

        for i in range(0, phi.shape[0], skip):
            for j in range(0, phi.shape[1], skip):
                plt.plot(xs=phi[i, j, :, 0].ravel('F'), ys=phi[i, j, :, 1].ravel('F'), zs=phi[i, j, :, 2].ravel('F'), **kwargs)

        # Make sure the upper, right and rear boundaries are plotted
        if (phi.shape[0] - 1) % skip != 0 and (phi.shape[1] - 1) % skip != 0 and (phi.shape[2] - 1) % skip != 0:
            for j in range(0, phi.shape[2], skip):
                plt.plot(xs=phi[-1, :, j, 0].ravel('F'), ys=phi[-1, :, j, 1].ravel('F'), zs=phi[-1, :, j, 2].ravel('F'), **kwargs)
                plt.plot(xs=phi[:, -1, j, 0].ravel('F'), ys=phi[:, -1, j, 1].ravel('F'), zs=phi[:, -1, j, 2].ravel('F'), **kwargs)
            for j in range(0, phi.shape[0], skip):
                plt.plot(xs=phi[j, :, -1, 0].ravel('F'), ys=phi[j, :, -1, 1].ravel('F'), zs=phi[j, :, -1, 2].ravel('F'), **kwargs)
                plt.plot(xs=phi[j, -1, :, 0].ravel('F'), ys=phi[j, -1, :, 1].ravel('F'), zs=phi[j, -1, :, 2].ravel('F'), **kwargs)
            for j in range(0, phi.shape[1], skip):
                plt.plot(xs=phi[:, j, -1, 0].ravel('F'), ys=phi[:, j, -1, 1].ravel('F'), zs=phi[:, j, -1, 2].ravel('F'), **kwargs)
                plt.plot(xs=phi[-1, j, :, 0].ravel('F'), ys=phi[-1, j, :, 1].ravel('F'), zs=phi[-1, j, :, 2].ravel('F'), **kwargs)

    @staticmethod
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    @staticmethod
    def scrollView3(volume, omega=None, permute=None, colormap='gray', colorbar=False):
        if permute is not None:
            volume = np.transpose(volume, permute)

        plot.remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        fig, ax = plt.subplots()
        ax.vol = volume
        ax.idx = volume.shape[-1] // 2
        ax.viewer = 'scrollView3'
        cl = ax.imshow(volume[:,:,ax.idx],cmap=colormap, extent=omega,norm=mpl.colors.Normalize(vmin=np.min(volume),vmax=np.max(volume)))
        fig.colorbar(cl) if colorbar else 1
        ttl = 'Position: {0} of {1}'.format(ax.idx,volume.shape[-1])
        ax.set_xlabel(ttl)
        fig.canvas.mpl_connect('key_press_event', plot.process_key)

    @staticmethod
    def scrollView4(volume, omega=None, permute=None):
        if permute is not None:
            volume = np.transpose(volume, permute)
            if omega is not None:
                if len(omega) < len(volume.shape):
                    print('Warning: Omega should be provided in full length!')
                omega = omega[np.array([1,3,5,7])]
                omega = omega[np.array(permute)]
                omega = omega[0:2]
                omega = utils.get_omega(omega)

        plot.remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        fig, ax = plt.subplots()
        ax.vol = volume
        ax.idx = volume.shape[-2] // 2
        ax.tpnt = volume.shape[-1] // 2
        ax.viewer = 'scrollView4'
        ax.imshow(volume[:,:,ax.idx,ax.tpnt], cmap='gray', extent=omega)
        fig.canvas.mpl_connect('key_press_event', plot.process_key)

    @staticmethod
    def scrollGrid2(gridStack):
        plot.remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        fig, ax = plt.subplots()
        ax.vol = gridStack
        ax.idx = gridStack.shape[-1] // 2
        ax.viewer = 'scrollGrid2'
        plot.plotGrid(gridStack[:, :, :, ax.idx])
        fig.canvas.mpl_connect('key_press_event', plot.process_key)

    @staticmethod
    def scrollGrid3(gridStack, skip=None, newFig=True):
        plot.remove_keymap_conflicts({'w', 'a', 's', 'd', 'up', 'down', 'left', 'right'})
        fig, ax = plt.subplots()
        ax.vol = gridStack
        ax.idx = gridStack.shape[-1] // 2
        ax.viewer = 'scrollGrid3'
        plot.plotGrid3(gridStack[:, :, :, :, ax.idx], skip=skip, newFig=newFig)
        fig.canvas.mpl_connect('key_press_event', plot.process_key)

    @staticmethod
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'w' or event.key == 'up': # arrow up
            plot.previous_slice(ax)
        elif event.key == 's' or event.key == 'down': # arrow down
            plot.next_slice(ax)
        elif (event.key == 'a' or event.key == 'left') and ax.viewer == 'scrollView4': # arrow left
            plot.previous_timepoint(ax)
        elif (event.key == 'd' or event.key == 'right') and ax.viewer == 'scrollView4': # arrow right
            plot.next_timepoint(ax)
        fig.canvas.draw()

    @staticmethod
    def previous_slice(ax):
        vol = ax.vol
        if ax.viewer == 'scrollView3':
            ax.idx = (ax.idx - 1) % vol.shape[-1]
            ax.images[0].set_array(vol[:, :, ax.idx])
        elif ax.viewer == 'scrollView4':
            ax.idx = (ax.idx - 1) % vol.shape[-2]
            ax.images[0].set_array(vol[:,:,ax.idx,ax.tpnt])
        elif ax.viewer == 'scrollGrid2':
            ax.idx = (ax.idx - 1) % vol.shape[-1]
            plt.gcf(); plt.cla()
            plot.plotGrid(vol[:, :, :, ax.idx])
        elif ax.viewer == 'scrollGrid3':
            ax.idx = (ax.idx - 1) % vol.shape[-1]
            plt.gca(); plt.cla()
            plot.plotGrid3(vol[:, :, :, :, ax.idx], newFig=False)
        ttl = 'Position: {0} of {1}'.format(ax.idx, vol.shape[-1])
        ax.set_xlabel(ttl)

    @staticmethod
    def previous_timepoint(ax):
        vol = ax.vol
        ax.tpnt = (ax.tpnt - 1) % vol.shape[-1]
        ax.images[0].set_array(vol[:,:,ax.idx,ax.tpnt])

    @staticmethod
    def next_slice(ax):
        vol = ax.vol
        if ax.viewer == 'scrollView3':
            ax.idx = (ax.idx + 1) % vol.shape[-1]
            ax.images[0].set_array(vol[:, :, ax.idx])
        elif ax.viewer == 'scrollView4':
            ax.idx = (ax.idx + 1) % vol.shape[-2]
            ax.images[0].set_array(vol[:,:,ax.idx,ax.tpnt])
        elif ax.viewer == 'scrollGrid2':
            ax.idx = (ax.idx + 1) % vol.shape[-1]
            plt.gcf(); plt.cla()
            plot.plotGrid(vol[:, :, :, ax.idx])
        elif ax.viewer == 'scrollGrid3':
            ax.idx = (ax.idx + 1) % vol.shape[-1]
            plt.gca(); plt.cla()
            plot.plotGrid3(vol[:, :, :, :, ax.idx], newFig=False)
        ttl = 'Position: {0} of {1}'.format(ax.idx, vol.shape[-1])
        ax.set_xlabel(ttl)

    @staticmethod
    def next_timepoint(ax):
        vol = ax.vol
        ax.tpnt = (ax.tpnt + 1) % vol.shape[-1]
        ax.images[0].set_array(vol[:,:,ax.idx,ax.tpnt])

    @staticmethod
    def plotMip(volume, omega=None, permute=None, colormap='gray', colorbar=False, fliptranspose=False):
        if permute is not None:
            volume = np.transpose(volume, permute)

        if fliptranspose:
            volume = np.flipud(volume.transpose((1,0,2)))

        fig, ax = plt.subplots()
        mip = np.max(volume,axis=2)
        cl = ax.imshow(mip,cmap=colormap, extent=omega)#,norm=mpl.colors.Normalize(vmin=np.min(volume),vmax=np.max(volume)))
        fig.colorbar(cl) if colorbar else 1