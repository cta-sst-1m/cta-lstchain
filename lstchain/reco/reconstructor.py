from abc import abstractmethod, ABC
import inspect
import logging

from iminuit import Minuit
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import astropy.units as u

from ctapipe.reco.reco_algorithms import Reconstructor
from lstchain.io.lstcontainers import DL1ParametersContainer
from lstchain.image.pdf import log_gaussian, log_gaussian2d
from lstchain.visualization.camera import display_array_camera

logger = logging.getLogger(__name__)

range_ext = 5

class DL0Fitter(ABC):
    """
        Base class for the extraction of DL1 parameters from R1 events
        using a log likelihood minimisation method.

    """

    def __init__(self, waveform, error, sigma_s, geometry, dt, n_samples, start_parameters,
                 template, gain=1, is_high_gain=0, baseline=0, crosstalk=0,
                 sigma_space=4, sigma_time=3,
                 time_before_shower=10, time_after_shower=50,
                 bound_parameters=None):
        """
            Initialise the data and parameters used for the fitting.

            Parameters:
            -----------
            waveform: array
                Calibrated signal in each pixel versus time
            error: array or None
                Error on the waveform
            sigma_s: array
                Standard deviation of the amplitude of the single photo-electron pluse
            geometry:
                Camera geometry
            dt: float
                Duration of time samples
            n_samples: int
                Number of time samples
            start_parameters: dictionary
                Starting value of the image parameters for the fit
            template: array
                Temporal template of the response of the pixel to a photo-electron
                Can contain two templates for high and low gain
            gain: array
                Selected gain in each pixel
            is_high_gain: boolean array
                Identify pixel with high gain selected
            baseline: array
                Remaining baseline in each pixel
            crosstalk: float
                Probability of a photo-electron to interact twice in a pixel
            sigma_space: float
                Size of the region over which the likelihood needs to be estimated
                in number of standard deviation away from the center of the spatial model
            sigma_time: float
                Time window around the peak of signal over which to compute
                the likelihood in number of temporal width of the signal
            time_before_shower: float
                Duration before the start of the signal which is not ignored
            time_after_shower:
                Duration after the end of the signal which is not ignored
            bound_parameters: dictionary
                Bounds for the parameters used during the fit
        """

        self.gain = gain
        self.is_high_gain = is_high_gain
        self.baseline = baseline
        self.sigma_s = sigma_s
        self.crosstalk = crosstalk

        self.geometry = geometry
        self.dt = dt
        self.template = template

        self.n_pixels, self.n_samples = len(geometry.pix_area), n_samples
        self.times = np.arange(0, self.n_samples) * self.dt

        self.pix_x = geometry.pix_x.value
        self.pix_y = geometry.pix_y.value
        self.pix_area = geometry.pix_area.to(u.m**2).value

        self.labels = {'charge': 'Charge [p.e.]',
                       't_cm': '$t_{CM}$ [ns]',
                       'x_cm': '$x_{CM}$ [m]',
                       'y_cm': '$y_{CM}$ [m]',
                       'width': '$\sigma_w$ [m]',
                       'length': '$\sigma_l$ [m]',
                       'psi': '$\psi$ [rad]',
                       'v': '$v$ [m/ns]'
                       }

        self.sigma_space = sigma_space
        self.sigma_time = sigma_time
        self.time_before_shower = time_before_shower
        self.time_after_shower = time_after_shower

        self.start_parameters = start_parameters
        self.bound_parameters = bound_parameters
        self.end_parameters = None
        self.names_parameters = list(inspect.signature(self.log_pdf).parameters)
        self.error_parameters = None
        self.correlation_matrix = None

        self.mask_pixel, self.mask_time = self.clean_data()
        self.gain = gain[self.mask_pixel]
        self.is_high_gain = is_high_gain[self.mask_pixel]
        self.baseline = baseline[self.mask_pixel]
        self.sigma_s = sigma_s[self.mask_pixel]
        self.crosstalk = crosstalk[self.mask_pixel]

        self.n_pixels, self.n_samples = len(geometry.pix_area), n_samples
        self.times = (np.arange(0, self.n_samples) * self.dt)[self.mask_time]

        self.pix_x = geometry.pix_x.value[self.mask_pixel]
        self.pix_y = geometry.pix_y.value[self.mask_pixel]
        self.pix_area = geometry.pix_area.to(u.m**2).value[self.mask_pixel]

        self.data = waveform
        self.error = error

        filter_pixels = np.arange(self.n_pixels)[~self.mask_pixel]
        filter_times = np.arange(self.n_samples)[~self.mask_time]

        if error is None:

            std = np.std(self.data[~self.mask_pixel])
            self.error = np.ones(self.data.shape) * std

        self.data = np.delete(self.data, filter_pixels, axis=0)
        self.data = np.delete(self.data, filter_times, axis=1)
        self.error = np.delete(self.error, filter_pixels, axis=0)
        self.error = np.delete(self.error, filter_times, axis=1)

    @abstractmethod
    def clean_data(self):

        pass

    def __str__(self):

        str = 'Start parameters :\n\t{}\n'.format(self.start_parameters)
        str += 'Bound parameters :\n\t{}\n'.format(self.bound_parameters)
        str += 'End parameters :\n\t{}\n'.format(self.end_parameters)
        str += 'Error parameters :\n\t{}\n'.format(self.error_parameters)
        str += 'Log-Likelihood :\t{}'.format(self.log_likelihood(**self.end_parameters))

        return str

    def fill_event(self, data, error=None):
        """

        Parameters
        ----------
        data: DL0 waveforms (n_pixels, n_samples)
        error Associated errors to the DL0 waveforms (n_pixels, n_samples)
        Returns
        -------

        """
    def fit(self, verbose=True, minuit=True, ncall=None, **kwargs):
        """
            Performs the fitting procedure.

        Parameters
        ----------
        verbose: boolean
        minuit: boolean
            If True, minuit is used to perform the fit. Else, scipy optimize is used instead
        ncall: int
            Maximum number of call for migrad
        """

        if minuit:

            fixed_params = {}
            bounds_params = {}
            start_params = self.start_parameters

            if self.bound_parameters is not None:

                for key, val in self.bound_parameters.items():

                    bounds_params['limit_' + key] = val

            for key in self.names_parameters:

                if key in kwargs.keys():

                    fixed_params['fix_' + key] = True

                else:

                    fixed_params['fix_' + key] = False

            # print(fixed_params, bounds_params, start_params)
            options = {**start_params, **bounds_params, **fixed_params}
            f = lambda *args: -self.log_likelihood(*args)
            print_level = 2 if verbose else 0
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad(ncall=ncall)
            self.end_parameters = dict(m.values)
            options = {**self.end_parameters, **bounds_params, **fixed_params}
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad(ncall=ncall)
            try:
                self.error_parameters = dict(m.errors)

            except (KeyError, AttributeError, RuntimeError):

                self.error_parameters = {key: np.nan for key in self.names_parameters}
                pass
            # print(self.end_parameters, self.error_parameters)

        else:

            fixed_params = {}

            for param in self.names_parameters:
                if param in kwargs.keys():
                    fixed_params[param] = kwargs[param]
                    del kwargs[param]

            start_parameters = []
            bounds = []
            name_parameters = []

            for key in self.names_parameters:

                if key not in fixed_params.keys():

                    start_parameters.append(self.start_parameters[key])
                    bounds.append(self.bound_parameters[key])
                    name_parameters.append(key)

            def llh(x):

                params = dict(zip(name_parameters, x))
                return -self.log_likelihood(**params, **fixed_params)

            result = minimize(llh, x0=start_parameters, bounds=bounds, **kwargs)
            self.end_parameters = dict(zip(name_parameters, result.x))
            self.end_parameters.update(fixed_params)

            try:
                self.correlation_matrix = result.hess_inv.todense()
                self.error_parameters = dict(zip(name_parameters,
                                             np.diagonal(np.sqrt(self.correlation_matrix))))
            except (KeyError, AttributeError):
                pass

            if verbose:

                print(result)

    def pdf(self, *args, **kwargs):

        return np.exp(self.log_pdf(*args, **kwargs))

    @abstractmethod
    def plot(self):

        pass

    @abstractmethod
    def compute_bounds(self):

        pass

    @abstractmethod
    def log_pdf(self, *args, **kwargs):

        pass

    def likelihood(self, *args, **kwargs):

        return np.exp(self.log_likelihood(*args, **kwargs))

    def log_likelihood(self, *args, **kwargs):

        llh = self.log_pdf(*args, **kwargs)
        return np.sum(llh)

    def plot_1dlikelihood(self, parameter_name, axes=None, size=1000,
                          x_label=None, invert=False):

        key = parameter_name

        if key not in self.names_parameters:

            raise NameError('Parameter : {} not in existing parameters :'
                            '{}'.format(key, self.names_parameters))

        x = np.linspace(self.bound_parameters[key][0], self.bound_parameters[key][1], num=size)
        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        params = copy(self.end_parameters)
        llh = np.zeros(x.shape)

        for i, xx in enumerate(x):

            params[key] = xx
            llh[i] = self.log_likelihood(**params)

        x_label = self.labels[key] if x_label is None else x_label

        if not invert:
            axes.plot(x, -llh, color='r')
            axes.axvline(self.end_parameters[key], linestyle='--', color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axvline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            # axes.axvspan(self.bound_parameters[key][0],
            #              self.bound_parameters[key][1], label='bounds',
            #             alpha=0.5, facecolor='k')
            axes.set_ylabel('-$\ln \mathcal{L}$')
            axes.set_xlabel(x_label)

        else:

            axes.plot(-llh, x, color='r')
            axes.axhline(self.end_parameters[key], linestyle='--',
                         color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axhline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.axhspan(self.bound_parameters[key][0],
                         self.bound_parameters[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
            axes.set_xlabel('-$\ln \mathcal{L}$')
            axes.set_ylabel(x_label)
            axes.xaxis.set_label_position('top')
            # axes.xaxis.tick_top()

        axes.legend(loc='best')
        return axes

    def plot_2dlikelihood(self, parameter_1, parameter_2=None, size=100,
                          x_label=None, y_label=None):

        if isinstance(size, int):
            size = (size, size)

        key_x = parameter_1
        key_y = parameter_2
        x = np.linspace(self.bound_parameters[key_x], self.bound_parameters[key_x], num=size[0])
        y = np.linspace(self.bound_parameters[key_y], self.bound_parameters[key_y], num=size[1])
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        params = copy(self.end_parameters)
        llh = np.zeros(size)

        for i, xx in enumerate(x):
            params[key_x] = xx
            for j, yy in enumerate(y):

                params[key_y] = yy
                llh[i, j] = self.log_likelihood(**params)


        fig = plt.figure()
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_center = [left, bottom, width, height]
        rect_x = [left, bottom + height + spacing, width, 0.2]
        rect_y = [left + width + spacing, bottom, 0.2, height]
        axes = fig.add_axes(rect_center)
        axes_x = fig.add_axes(rect_x)
        axes_y = fig.add_axes(rect_y)
        axes.tick_params(direction='in', top=True, right=True)
        self.plot_1dlikelihood(parameter_name=parameter_1, axes=axes_x)
        self.plot_1dlikelihood(parameter_name=parameter_2, axes=axes_y,
                               invert=True)
        axes_x.tick_params(direction='in', labelbottom=False)
        axes_y.tick_params(direction='in', labelleft=False)

        axes_x.set_xlabel('')
        axes_y.set_ylabel('')
        axes_x.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
        axes_y.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

        x_label = self.labels[key_x] if x_label is None else x_label
        y_label = self.labels[key_y] if y_label is None else y_label

        im = axes.imshow(-llh.T,  origin='lower', extent=[x.min() - dx/2.,
                                                     x.max() - dx/2,
                                                     y.min() - dy/2,
                                                     y.max() - dy/2], aspect='auto')

        axes.scatter(self.end_parameters[key_x], self.end_parameters[key_y],
                     marker='x', color='w', label='Maximum Likelihood')
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(loc='best')
        plt.colorbar(mappable=im, ax=axes_y, label='-$\ln \mathcal{L}$')
        axes_x.set_xlim(x.min(), x.max())
        axes_y.set_ylim(y.min(), y.max())

        return axes

    def plot_likelihood(self, parameter_1, parameter_2=None,
                        axes=None, size=100,
                        x_label=None, y_label=None):

        if parameter_2 is None:

            return self.plot_1dlikelihood(parameter_name=parameter_1,
                                          axes=axes, x_label=x_label, size=size)

        else:

            return self.plot_2dlikelihood(parameter_1,
                                          parameter_2=parameter_2,
                                          size=size,
                                          x_label=x_label,
                                          y_label=y_label)


class TimeWaveformFitter(DL0Fitter, Reconstructor):
    """
        Specific class for the extraction of DL1 parameters from R1 events using a log likelihood minimisation method by fitting the spatial and temporal dependance of signal in the camera taking into account the calibrated response of the pixels.
    """


    def __init__(self, *args, image, n_peaks=100, **kwargs):
        """
            Initialise the data and parameters used for the fitting, including method specific objects.

            Parameters:
            *args, **kwargs: Argument for the DL0Fitter initialisation
            image: ADDTYPE
                Image of the event in the camera
            n_peak: int
                Maximum upper bound of the sum over possible detected photo-electron value in the likelihood computation.
            -----------
        """
        self.image = image
        super().__init__(*args, **kwargs)
        self._initialize_pdf(n_peaks=n_peaks)

    def clean_data(self):
        """
            Method used to select pixels and time samples used in the fitting procedure.
        """

        x_cm = self.start_parameters['x_cm']
        y_cm = self.start_parameters['y_cm']
        width = self.start_parameters['width']
        length = self.start_parameters['length']
        psi = self.start_parameters['psi']

        dx = self.pix_x - x_cm
        dy = self.pix_y - y_cm

        lon = dx * np.cos(psi) + dy * np.sin(psi)
        lat = dx * np.sin(psi) - dy * np.cos(psi)

        mask_pixel = ((lon / length) ** 2 + (
                    lat / width) ** 2) < self.sigma_space ** 2

        v = self.start_parameters['v']
        t_start = (self.start_parameters['t_cm']
                   - (np.abs(v) * length / 2 * self.sigma_time)
                   - self.time_before_shower)
        t_end = (self.start_parameters['t_cm']
                 + (np.abs(v) * length / 2 * self.sigma_time)
                 + self.time_after_shower)

        mask_time = (self.times < t_end) * (self.times > t_start)

        return mask_pixel, mask_time

    def _initialize_pdf(self, n_peaks):
        """
            Compute quantities used at each iteration of the fitting procedure.
        """
        # May need rework after accelaration addition
        photoelectron_peak = np.arange(range_ext*n_peaks, dtype=np.int)
        # the range_ext* is for testing only, extends the range available for the sum in the likelihood
        self.photo_peaks = photoelectron_peak
        photoelectron_peak = photoelectron_peak[..., None]
        mask = (self.photo_peaks == 0)
        self.photo_peaks[mask] = 1
        log_k = np.log(self.photo_peaks)
        log_k = np.cumsum(log_k)
        self.photo_peaks[mask] = 0
        self.log_k = log_k
        self.crosstalk_factor = photoelectron_peak * self.crosstalk
        self.crosstalk_factor = self.crosstalk_factor

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):
        """
            Compute the log likelihood of the model used for a set of input parameters.

        Parameters
        ----------
        self: Contains all the information about the data and calibration
        charge: float
            Charge of the peak of the spatial model
        t_cm: float
            Time of the middle of the energy deposit in the camera for the temporal model
        x_cm, y_cm: float
            Position of the center of the spatial model
        length, width: float
            Spatial dispersion of the model along the main and minor axis
        psi: float
            Orientation of the main axis of the spatial model and of the propagation of the temporal model
        v: float
            Velocity of the evolution of the signal over the camera
        """

        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., np.newaxis] - t
        t = t.T

        log_mu = log_gaussian2d(size=charge * self.pix_area,
                                x=self.pix_x,
                                y=self.pix_y,
                                x_cm=x_cm,
                                y_cm=y_cm,
                                width=width,
                                length=length,
                                psi=psi)
        mu = np.exp(log_mu)

        # We reduce the sum by limiting to the poisson term contributing for more than 10^6
        # The limits are approximated by 2 broken linear function obtained for 0 crosstalk
        # The choice of kmin and kmax is currently not done on a pixel basis
        kmin = np.zeros(len(mu))
        kmax = np.zeros(len(mu))
        it = 0
        for elt in mu:
            if elt < 120:
                kmin[it] = 0.66 * (elt-20)
                kmax[it] = 1.34 * (elt-20) + 45
            else:
                kmin[it] = 0.904 * elt - 42.8
                kmax[it] = 1.096 * elt + 47.8
            it = it + 1
        kmin[kmin<0] = 0
        kmax = np.ceil(kmax)
        mask = (kmax <= len(self.log_k) / range_ext)
        # /range_ext for testing only
        # compensate the extension of len(self.log_k) back to n_peak
        if len(kmin[mask]) == 0 or len(kmax[mask]) == 0:
            kmin, kmax = 0, len(self.log_k)
        else:
            #kmin, kmax = min(kmin[mask].astype(int)), max(kmax[mask].astype(int))
            kmin, kmax = min(kmin.astype(int)), max(kmax.astype(int))

        if kmax > len(self.log_k):
            kmax = len(self.log_k)
            logger.debug("kmax forced to %s", kmax)
            # Only usefull to compare the sum with the Gaussian approx.
            # Actual implementation should use n_peak as length and
            # compute only the gaussian approx for higher kmax

        log_k = self.log_k

        self.photo_peaks = np.arange(kmin, kmax, dtype=np.int)
        self.crosstalk_factor = self.photo_peaks[..., None]*self.crosstalk

        x = mu + self.crosstalk_factor
        log_x = np.log(x)

        log_x = ((self.photo_peaks - 1) * log_x.T).T
        log_poisson = log_mu - log_k[kmin:kmax][..., None] - x + log_x

        mean_LG = self.photo_peaks * ((self.gain[..., None] *
                                       self.template(t, gain='LG')))[..., None]

        mean_HG = self.photo_peaks * ((self.gain[..., None] *
                                       self.template(t, gain='HG')))[
            ..., None]

        mean = (mean_HG.T * self.is_high_gain) + mean_LG.T * (~self.is_high_gain)
        mean = mean.T

        x = self.data - self.baseline[..., None]

        sigma_n_LG = self.photo_peaks * ((self.sigma_s[..., None] *
                                          self.template(t, gain='LG')) ** 2)[..., None]

        sigma_n_HG = self.photo_peaks * ((self.sigma_s[..., None] *
                                          self.template(t, gain='HG')) ** 2)[..., None]

        sigma_n = sigma_n_HG.T * self.is_high_gain + sigma_n_LG.T * (~self.is_high_gain)
        sigma_n = sigma_n.T

        sigma_n = (self.error**2)[..., None] + sigma_n
        sigma_n = np.sqrt(sigma_n)


        if np.any(~mask):
            mu_hat_LG = ((mu[~mask] / (1-self.crosstalk_factor[~mask]))
                         * (self.gain[~mask][..., None] * self.template[~mask](t, gain='LG')))
            mu_hat_HG = ((mu[~mask] / (1-self.crosstalk_factor[~mask]))
                         * (self.gain[~mask][..., None] * self.template[~mask](t, gain='HG')))
            mu_hat = (mu_hat_HG.T * self.is_high_gain[~mask]) + mu_hat_LG.T * (~self.is_high_gain[~mask])
            mu_hat = mu_hat.T

            sigma_hat_LG = ((mu[~mask] / np.power(1-self.crosstalk_factor[~mask], 3))
                            * (self.gain[~mask][..., None]
                               * self.template[~mask](t, gain='LG')) ** 2)
            sigma_hat_HG = ((mu[~mask] / np.power(1-self.crosstalk_factor[~mask], 3))
                            * (self.gain[~mask][..., None]
                               * self.template[~mask](t, gain='HG')) ** 2)
            sigma_hat = (sigma_hat_HG.T * self.is_high_gain[~mask]) + sigma_hat_LG.T * (~self.is_high_gain[~mask])
            sigma_hat = sigma_hat.T
            sigma_hat = np.sqrt((self.error[~mask]**2)[..., None] + sigma_hat)

            log_pdf_HL = log_gaussian(x[~mask][..., None], mu_hat, sigma_hat)


        log_gauss = log_gaussian(x[..., None], mean, sigma_n)
        log_poisson = np.expand_dims(log_poisson.T, axis=1)
        log_pdf = log_poisson + log_gauss
        pdf = np.sum(np.exp(log_pdf), axis=-1)

        log_pdf2 = 0
        if np.any(~mask):
            log_pdf = np.log(pdf)
            logger.debug("Gaussian approx %s", log_pdf_HL)
            logger.debug("Poisson sum %s", log_pdf[~mask])
            logger.debug("diff %s", log_pdf_HL-log_pdf[~mask])
            pdf2 = pdf[mask]
            mask = (pdf2 <= 0)
            pdf2 = pdf2[~mask]
            n_points_LL = pdf2.size
            n_points_HL = log_pdf_HL.size()
            log_pdf2 = ((np.log(pdf2).sum() + log_pdf_HL.sum())
                        / (n_points_LL + n_points_HL))


        mask = (pdf <= 0)
        pdf = pdf[~mask]
        n_points = pdf.size
        log_pdf = np.log(pdf).sum() / n_points

        logger.debug("Final pdf %s", log_pdf)
        logger.debug("Final pdf with Gaussian approx %s", log_pdf2)

        return log_pdf

    def predict(self, container=DL1ParametersContainer(), **kwargs):
        """
            Call the fitting procedure and fill the results.

        Parameters
        ----------
        container: DL1ParametersContainer
            Location to fill with updated DL1 parameters
        """


        self.fit(**kwargs)

        container.x = self.end_parameters['x_cm'] * u.m
        container.y = self.end_parameters['y_cm'] * u.m
        container.r = np.sqrt(container.x**2 + container.y**2)
        container.phi = np.arctan2(container.y, container.x)
        container.psi = self.end_parameters['psi'] * u.rad # TODO turn psi angle when length < width
        container.length = max(self.end_parameters['length'], self.end_parameters['width']) * u.m
        container.width = min(self.end_parameters['length'], self.end_parameters['width']) * u.m

        container.time_gradient = self.end_parameters['v']
        container.intercept = self.end_parameters['t_cm']

        container.wl = container.width / container.length

        return container

    def compute_bounds(self):
        pass

    def plot(self, n_sigma=3, init=False):

        charge = self.image
        cam_display = display_array_camera(charge, geom=self.geometry,)

        if init:
            params = self.start_parameters
        else:
            params = self.end_parameters

        length = n_sigma * params['length']
        psi = params['psi']
        dx = length * np.cos(psi)
        dy = length * np.sin(psi)
        """
        direction_arrow = Arrow(x=params['x_cm'],
                                y=params['y_cm'],
                                dx=dx, dy=dy, width=10, color='k',
                                label='EAS direction')

        cam_display.axes.add_patch(direction_arrow)
        """

        cam_display.add_ellipse(centroid=(params['x_cm'],
                                        params['y_cm']),
                                width=n_sigma * params['width'],
                                length=length,
                                angle=psi,
                                linewidth=6, color='r', linestyle='--',
                                label='{} $\sigma$ contour'.format(n_sigma))
        cam_display.axes.legend(loc='best')

        return cam_display

    def plot_waveforms(self, axes=None):

        image = self.image[self.mask_pixel]
        n_pixels = min(15, len(image))
        pixels = np.argsort(image)[-n_pixels:]
        dx = (self.pix_x[pixels] - self.end_parameters['x_cm'])
        dy = (self.pix_y[pixels] - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(
            self.end_parameters['psi'])
        fitted_times = np.polyval(
            [self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)
        # fitted_times = fitted_times[pixels]
        times_index = np.argsort(fitted_times)

        # waveforms = self.data[times_index]
        waveforms = self.data[pixels]
        waveforms = waveforms[times_index]
        long_pix = long_pix[times_index]
        fitted_times = fitted_times[times_index]

        X, Y = np.meshgrid(self.times, long_pix)

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        M = axes.pcolormesh(X, Y, waveforms)
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('Longitude [m]')
        label = self.labels['t_cm'] + ' : {:.2f} [ns]'.format(self.end_parameters['t_cm'])
        label += '\n' + self.labels['v'] + ' : {:.2f} [m/ns]'.format(self.end_parameters['v'])
        axes.plot(fitted_times, long_pix, color='w',
                 label=label)
        axes.legend(loc='best')
        axes.get_figure().colorbar(label='[LSB]', ax=axes, mappable=M)

        return axes



class NormalizedPulseTemplate:
    """
        Class for handling the template for the pulsed response of the pixels of the camera to a single photo-electron in high and low gain.
    """

    def __init__(self, amplitude_HG, amplitude_LG, time, amplitude_HG_std=None, amplitude_LG_std=None):
        """
            Save the pulse template and optionnal error and create an interpolation.

        Parameters
        ----------
        amplitude_HG/LG: array
            Amplitude of the signal produced in a pixel by a photo-electron in high gain (HG) and low gain (LG) for successive time samples
        time: array
            Times of the samples
        amplitude_HG/LG_std: array
            Error on the pulse template amplitude
        """
        self.time = np.array(time)
        self.amplitude_HG = np.array(amplitude_HG)
        self.amplitude_LG = np.array(amplitude_LG)  
        if amplitude_HG_std is not None:
            assert np.array(amplitude_HG_std).shape == self.amplitude_HG.shape
            self.amplitude_HG_std = np.array(amplitude_HG_std)
        else:
            self.amplitude_HG_std = np.zeros(self.amplitude_HG.shape)
        if amplitude_LG_std is not None:
            assert np.array(amplitude_LG_std).shape == self.amplitude_LG.shape
            self.amplitude_LG_std = np.array(amplitude_LG_std)
        else:
            self.amplitude_LG_std = self.amplitude_LG * 0
        self._template = self._interpolate()
        self._template_std = self._interpolate_std()

    def __call__(self, time, gain, amplitude=1, t_0=0, baseline=0):
        """
            Use the interpolated template to access the value of the pulse at time = time in gain regime = gain. Additionnally, an alternative normalisation, origin of time and baseline can be used.

        Parameters
        ----------
        time: float array
            Time after the origin after which to estimate the value of the pulse
        gain: string array
            Identifier of the gain channel used for each pixel. Either "HG" or "LG"
        amplitude: float
            Normalisation factor to aply to the template
        t_0: float
            Shift in the origin of time
        baseline: float array
            Baseline to be substracted for each pixel

        Return
        ----------
        y: array
            Value of the template in each pixel at the requested times
        """
        y = amplitude * self._template[gain](time - t_0) + baseline
        return np.array(y)

    def std(self, time, amplitude=1, t_0=0, baseline=0):
        """

        """
        y = amplitude * self._template_std(time - t_0) + baseline
        return np.array(y)
    """
    def __getitem__(self, item):
        return NormalizedPulseTemplate(amplitude=self.amplitude[item],
                                       time=self.time)
    """
    def save(self, filename):
        """
            Save a loaded template to a text file.
        """
        data = np.vstack([self.time, self.amplitude_HG, self.amplitude_HG_std, self.amplitude_LG, self.amplitude_LG_std])
        np.savetxt(filename, data.T)

    @classmethod
    def load(cls, filename):
        """
        Load a pulse template from a text file. Allows for only one gain channel and no errors, two gain channels and no errors or two gain channels with errors.

        Parameters
        ----------
        cls: This class
        filename: string
            Location of the template file

        Return
        ----------
        cls(): Instance of NormalizedPulseTemplate receiving the information from the input file
        """
        data = np.loadtxt(filename).T
        assert len(data) in [2, 3, 5]
        if len(data) == 2:  # one shape in file
            t, x = data
            return cls(amplitude_HG=x, amplitude_LG=x, time=t)
        if len(data) == 3:  # no error in file
            t, hg, lg = data
            return cls(amplitude_HG=hg, amplitude_LG=lg, time=t)
        elif len(data) == 5: # two gains and errors
            t, hg, lg, dhg, dlg = data
            return cls(amplitude_HG=hg, amplitude_LG=lg, time=t, amplitude_HG_std=dhg, amplitude_LG_std=dlg)


    def _interpolate(self):
        """
        Creates a normalised interpolation of the pulse template from a discrete and non-normalised input. Also normalises the error.

        Return
        ----------
        A dictionnary containning a 1d cubic interpolation of the normalised amplitude of the template versus time, for the high and low gain channels.
        """
        if abs(np.min(self.amplitude_HG)) <= abs(np.max(self.amplitude_HG)):
                normalization = np.max(self.amplitude_HG)
        else:
            normalization = np.min(self.amplitude_HG)

        self.amplitude_HG = self.amplitude_HG / normalization
        self.amplitude_HG_std = self.amplitude_HG_std / normalization

        if abs(np.min(self.amplitude_LG)) <= abs(np.max(self.amplitude_LG)):
            normalization = np.max(self.amplitude_LG)
        else:
            normalization = np.min(self.amplitude_LG)

        self.amplitude_LG = self.amplitude_LG / normalization
        self.amplitude_LG_std = self.amplitude_LG_std / normalization

        return {"HG" : interp1d(self.time, self.amplitude_HG, kind='cubic',
                                bounds_error=False, fill_value=0., assume_sorted=True),
                "LG" : interp1d(self.time, self.amplitude_LG, kind='cubic',
                                bounds_error=False, fill_value=0., assume_sorted=True)}
                

    def _interpolate_std(self):
        """
        Creates a normalised interpolation of the error on the pulse template from a discrete and normalised input.

        Return
        ----------
        A dictionnary containning a 1d cubic interpolation of the error on the normalised amplitude of the template versus time, for the high and low gain channels.
        """
        return {"HG" : interp1d(self.time, self.amplitude_HG_std, kind='cubic',
                       bounds_error=False, fill_value=np.inf, assume_sorted=True),
                "LG" : interp1d(self.time, self.amplitude_LG_std, kind='cubic',
                       bounds_error=False, fill_value=np.inf, assume_sorted=True)}

 
    # commenting unused old function for now
    """
    def integral(self, order=1):
        return np.trapz(y=self.amplitude**order, x=self.time)

    def compute_charge_amplitude_ratio(self, integral_width, dt_sampling):

        dt = self.time[1] - self.time[0]

        if not dt % dt_sampling:
            raise ValueError('Cannot use sampling rate {} for {}'.format(
                1 / dt_sampling, dt))
        step = int(dt_sampling / dt)
        y = self.amplitude[::step]
        window = np.ones(integral_width)
        charge_to_amplitude_factor = np.convolve(y, window)
        charge_to_amplitude_factor = np.max(charge_to_amplitude_factor)

        return 1 / charge_to_amplitude_factor

    def plot(self, axes=None, label='Template data-points', **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        t = np.linspace(self.time.min(), self.time.max(),
                        num=len(self.time) * 100)
        axes.plot(self.time, self.amplitude,
                          label=label, **kwargs)
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('normalised amplitude [a.u.]')
        axes.legend(loc='best')
        return axes

    def plot_interpolation(self, axes=None, sigma=-1., color='k',
                           label='Template interpolation', cumulative=False,
                           **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        t = np.linspace(self.time.min(), self.time.max(),
                        num=len(self.time) * 100)
        mean_y = self.__call__(t)
        std_y = self.std(t)
        if sigma > 0:
            axes.fill_between(
                t,
                mean_y + sigma * std_y,
                mean_y - sigma * std_y,
                alpha=0.3, color=color, label=label
            )
            axes.plot(t, mean_y, '-', color=color, **kwargs)
        else:
            axes.plot(t, mean_y, '-', label=label, color=color, **kwargs)
        if cumulative:
            integ = np.cumsum(mean_y)
            axes.plot(t, 1 - integ/integ[-1], '--', color=color,
                      label=label + ' integrated', **kwargs)
        axes.legend(loc='best')
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('normalised amplitude [a.u.]')
        return axes
    """
    def compute_time_of_max(self):
        """

        """
        t_max = (self.time[np.argmax(self.amplitude_HG)] +
                 self.time[np.argmax(self.amplitude_LG)])/2
        return t_max
        # old method below, more precise possible for one gain
        """
        dt = np.diff(self.time)[0]
        index_max = np.argmax(self.amplitude)
        t = np.linspace(self.time[index_max] - dt,
                        self.time[index_max] + dt,
                        num=100)
        t_max = self(t).argmax()
        t_max = t[t_max]
        return t_max
        """

