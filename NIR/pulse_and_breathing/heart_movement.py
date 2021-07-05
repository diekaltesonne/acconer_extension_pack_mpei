import numpy as np
import pyqtgraph as pg
from numpy import pi
from scipy.signal import butter, sosfilt

from PyQt5 import QtCore

from acconeer.exptool import configs, utils
from acconeer.exptool.clients import SocketClient, SPIClient, UARTClient
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess
from acconeer.exptool.structs import configbase


def main():
    args = utils.ExampleArgumentParser(num_sens=1).parse_args()
    utils.config_logging(args)

    if args.socket_addr:
        client = SocketClient(args.socket_addr)
    elif args.spi:
        client = SPIClient()
    else:
        port = args.serial_port or utils.autodetect_serial_port()
        client = UARTClient(port)

    sensor_config = get_sensor_config()
    processing_config = get_processing_config()
    sensor_config.sensor = args.sensors

    session_info = client.setup_session(sensor_config)

    pg_updater = PGUpdater(sensor_config, processing_config, session_info)
    pg_process = PGProcess(pg_updater)
    pg_process.start()

    client.start_session()

    interrupt_handler = utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    processor = HeartProcessor(sensor_config, processing_config, session_info)

    while not interrupt_handler.got_signal:
        info, data = client.get_next()
        plot_data = processor.process(data)

        if plot_data is not None:
            try:
                pg_process.put_data(plot_data)
            except PGProccessDiedException:
                break

    print("Disconnecting...")
    pg_process.close()
    client.disconnect()


def get_sensor_config():
    config = configs.IQServiceConfig()
    config.range_interval = [0.3, 0.8]
    config.update_rate = 80
    config.gain = 0.5
    config.repetition_mode = configs.IQServiceConfig.RepetitionMode.SENSOR_DRIVEN
    return config


class ProcessingConfiguration(configbase.ProcessingConfig):
    VERSION = 1

    hist_plot_len = configbase.FloatParameter(
        label="Plot length",
        unit="s",
        default_value=10,
        limits=(1, 30),
        decimals=0,
    )


get_processing_config = ProcessingConfiguration


class HeartProcessor:
    peak_hist_len = 600

    phase_weights_alpha = 0.9
    peak_loc_alpha = 0.95
    sweep_alpha = 0.7
    env_alpha = 0.95

    def __init__(self, sensor_config, processing_config, session_info):
        self.config = sensor_config

        assert sensor_config.update_rate is not None

        self.f = sensor_config.update_rate
        self.hist_plot_len = int(round(processing_config.hist_plot_len * self.f))
        self.breath_hist_len = max(2000, self.hist_plot_len)

        self.peak_history = np.zeros(self.peak_hist_len, dtype="complex")
        self.movement_history = np.zeros(self.peak_hist_len, dtype="float")
        self.breath_history = np.zeros(self.breath_hist_len, dtype="float")
        self.pulse_history = np.zeros(self.hist_plot_len, dtype="float")

        self.breath_sos = np.concatenate(butter(2, 2 * 0.3 / self.f))
        self.breath_zi = np.zeros((1, 2))

        #self.pulse_sos = np.concatenate(butter(2, 2 * np.array([5]) / self.f/10,btype='highpass',output='sos'))
        #self.pulse_sos = np.concatenate(butter(2, 2 * np.array([5]) / self.f/10,btype='highpass',output='sos'))
        self.pulse_sos = (butter(4, 2 * np.array([5]) / self.f/5,btype='highpass',output='sos'))
        print(self.pulse_sos)
        print(self.pulse_sos.shape)
        #self.pulse_sos = np.concatenate(butter(2, 2 * np.array([5]) / self.f/10,btype='highpass',output='sos'))
        
        #self.pulse_zi = np.zeros((1, 2))# !!!!
        self.pulse_zi = np.zeros((2, 2))# !!!!

        self.last_lp_sweep = None
        self.lp_phase_weights = None
        self.lp_sweep = None
        self.lp_peak_loc = 0

        self.sweep_index = 0

    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

    def process(self, sweep):
        if self.sweep_index == 0:
            self.lp_sweep = np.array(sweep)
            self.lp_env = np.abs(sweep)
            self.lp_peak_loc = np.argmax(self.lp_env)
            out_data = None
        else:
            self.lp_sweep = self.lp(sweep, self.lp_sweep, self.sweep_alpha)
            env = np.abs(self.lp_sweep)
            self.lp_env = self.lp(env, self.lp_env, self.env_alpha)
            peak_loc = np.argmax(self.lp_env)
            self.lp_peak_loc = self.lp(peak_loc, self.lp_peak_loc, self.peak_loc_alpha)

            peak_idx = int(round(self.lp_peak_loc))
            peak = np.mean(self.lp_sweep[peak_idx - 50 : peak_idx + 50])
            self.push(peak, self.peak_history)

            delta = self.lp_sweep * np.conj(self.last_lp_sweep)

            phase_weights = np.imag(delta)
            if self.lp_phase_weights is None:
                self.lp_phase_weights = phase_weights
            else:
                self.lp_phase_weights = self.lp(
                    phase_weights, self.lp_phase_weights, self.phase_weights_alpha)

            weights = np.abs(self.lp_phase_weights) * env

            delta_dist = np.dot(weights, np.angle(delta))
            delta_dist *= 2.5 / (2.0 * pi * sum(weights + 0.00001))

            y = self.movement_history[0] + delta_dist
            self.push(y, self.movement_history)

            y_pulse, self.pulse_zi = sosfilt(self.pulse_sos, np.array([y]), zi=self.pulse_zi)
            self.push(y_pulse, self.pulse_history)

            heart_hist_plot = self.pulse_history[: self.hist_plot_len]
            heart_hist_plot = np.array(np.flip(heart_hist_plot, axis=0))
            heart_hist_plot -= (np.max(heart_hist_plot) + np.min(heart_hist_plot)) * 0.5
            #heart_hist_plot -= (max(heart_hist_plot) + min(heart_hist_plot)) * 0.5

            out_data = {
                "peak_hist": self.peak_history[: 100],
                "peak_std_mm": 2.5 * np.std(np.unwrap(np.angle(self.peak_history))) / (2.0 * pi),
                "env_ampl": abs(self.lp_sweep),
                "env_delta": self.lp_phase_weights,
                "peak_idx": peak_idx,
                "heart_hist": heart_hist_plot,
            }

        self.last_lp_sweep = self.lp_sweep
        self.sweep_index += 1
        return out_data

    def lp(self, new, state, alpha):
        return alpha * state + (1 - alpha) * new

    def push(self, val, arr):
        res = np.empty_like(arr)
        res[0] = val
        res[1 :] = arr[: -1]
        arr[...] = res

class PGUpdater:
    def __init__(self, sensor_config, processing_config, session_info):
        assert sensor_config.update_rate is not None

        f = sensor_config.update_rate
        self.depths = utils.get_range_depths(sensor_config, session_info)
        self.hist_plot_len_s = processing_config.hist_plot_len
        self.hist_plot_len = int(round(self.hist_plot_len_s * f))
        self.move_xs = (np.arange(-self.hist_plot_len, 0) + 1) / f
        self.smooth_max = utils.SmoothMax(f, hysteresis=0.4, tau_decay=1.5)

    def setup(self, win):
        win.setWindowTitle("Выделение кардио составляющей")
        win.resize(800, 600)

        self.env_plot = win.addPlot(title="Amplitude of IQ data and change")
        self.env_plot.setMenuEnabled(False)
        self.env_plot.setMouseEnabled(x=False, y=False)
        self.env_plot.hideButtons()
        self.env_plot.addLegend()
        self.env_plot.showGrid(x=True, y=True)
        self.env_curve = self.env_plot.plot(
            pen=utils.pg_pen_cycler(0),
            name="Amplitude of IQ data",
        )
        self.delta_curve = self.env_plot.plot(
            pen=utils.pg_pen_cycler(1),
            name="Phase change between sweeps",
        )
        self.peak_vline = pg.InfiniteLine(pen=pg.mkPen("k", width=2.5, style=QtCore.Qt.DashLine))
        self.env_plot.addItem(self.peak_vline)

        self.peak_plot = win.addPlot(title="Phase of IQ at peak")
        self.peak_plot.setMenuEnabled(False)
        self.peak_plot.setMouseEnabled(x=False, y=False)
        self.peak_plot.hideButtons()
        utils.pg_setup_polar_plot(self.peak_plot, 1)
        self.peak_curve = self.peak_plot.plot(pen=utils.pg_pen_cycler(0))
        self.peak_scatter = pg.ScatterPlotItem(brush=pg.mkBrush("k"), size=15)
        self.peak_plot.addItem(self.peak_scatter)
        self.peak_text_item = pg.TextItem(color=pg.mkColor("k"), anchor=(0, 1))
        self.peak_plot.addItem(self.peak_text_item)
        self.peak_text_item.setPos(-1.15, -1.15)
        win.nextRow()

        # Heart rate
        self.heart_plot = win.addPlot(title="Heart movement")
        self.heart_plot.setMenuEnabled(False)
        self.heart_plot.setMouseEnabled(x=False, y=False)
        self.heart_plot.hideButtons()
        self.heart_plot.showGrid(x=True, y=True)
        self.heart_plot.setLabel("bottom", "Time (s)")
        self.heart_plot.setLabel("left", "Movement (mm)")
        self.heart_plot.setYRange(-0.5, 0.5)
        self.heart_plot.setXRange(-self.hist_plot_len_s, 0)
        self.heart_curve = self.heart_plot.plot(pen=utils.pg_pen_cycler(0))


    def update(self, data):
        envelope = data["env_ampl"]
        m = self.smooth_max.update(envelope)
        plot_delta = data["env_delta"] * m * 2e-5 + 0.5 * m

        norm_peak_hist_re = np.real(data["peak_hist"]) / m
        norm_peak_hist_im = np.imag(data["peak_hist"]) / m
        peak_std_text = "Std: {:.3f}mm".format(data["peak_std_mm"])
        peak_x = self.depths[data["peak_idx"]]

        self.env_plot.setYRange(0, m)
        self.env_curve.setData(self.depths, envelope)
        self.delta_curve.setData(self.depths, plot_delta)

        self.peak_scatter.setData([norm_peak_hist_re[0]], [norm_peak_hist_im[0]])
        self.peak_curve.setData(norm_peak_hist_re, norm_peak_hist_im)
        self.peak_text_item.setText(peak_std_text)
        self.peak_vline.setValue(peak_x)

        self.heart_curve.setData(self.move_xs, data["heart_hist"])

if __name__ == "__main__":
    main()
