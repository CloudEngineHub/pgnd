from pathlib import Path
from tqdm import tqdm
import time
import math
import psutil
import random
import fcntl
import os
import subprocess
import tempfile
from random import randint
from errno import EACCES

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import pyvista as pv

from pgnd.utils import get_root, mkdir
from pgnd.ffmpeg import make_video

root: Path = get_root(__file__)


class Xvfb(object):

    # Maximum value to use for a display. 32-bit maxint is the
    # highest Xvfb currently supports
    MAX_DISPLAY = 2147483647
    SLEEP_TIME_BEFORE_START = 0.1

    def __init__(
            self, width=800, height=680, colordepth=24,
            tempdir=None, display=None, **kwargs):
        self.width = width
        self.height = height
        self.colordepth = colordepth
        self._tempdir = tempdir or tempfile.gettempdir()
        self.new_display = display

        if not self.xvfb_exists():
            msg = (
                'Can not find Xvfb. Please install it with:\n'
                '   sudo apt install libgl1-mesa-glx xvfb')
            raise EnvironmentError(msg)

        self.extra_xvfb_args = ['-screen', '0', '{}x{}x{}'.format(
                                self.width, self.height, self.colordepth)]

        for key, value in kwargs.items():
            self.extra_xvfb_args += ['-{}'.format(key), value]

        if 'DISPLAY' in os.environ:
            self.orig_display = os.environ['DISPLAY'].split(':')[1]
        else:
            self.orig_display = None

        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.new_display is not None:
            if not self._get_lock_for_display(self.new_display):
                raise ValueError(f'Could not lock display :{self.new_display}')
        else:
            self.new_display = self._get_next_unused_display()
        display_var = ':{}'.format(self.new_display)
        self.xvfb_cmd = ['Xvfb', display_var] + self.extra_xvfb_args
        with open(os.devnull, 'w') as fnull:
            self.proc = subprocess.Popen(
                self.xvfb_cmd, stdout=fnull, stderr=fnull, close_fds=True)
        # give Xvfb time to start
        time.sleep(self.__class__.SLEEP_TIME_BEFORE_START)
        ret_code = self.proc.poll()
        if ret_code is None:
            self._set_display_var(self.new_display)
        else:
            self._cleanup_lock_file()
            raise RuntimeError(
                f'Xvfb did not start ({ret_code}): {self.xvfb_cmd}')

    def stop(self):
        try:
            if self.orig_display is None:
                del os.environ['DISPLAY']
            else:
                self._set_display_var(self.orig_display)
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait()
                except OSError:
                    pass
                self.proc = None
        finally:
            self._cleanup_lock_file()

    def xvfb_exists(self):
        """Check that Xvfb is available on PATH and is executable."""
        paths = os.environ['PATH'].split(os.pathsep)
        return any(os.access(os.path.join(path, 'Xvfb'), os.X_OK)
                   for path in paths)

    def _cleanup_lock_file(self):
        '''
        This should always get called if the process exits safely
        with Xvfb.stop() (whether called explicitly, or by __exit__).
        If you are ending up with /tmp/X123-lock files when Xvfb is not
        running, then Xvfb is not exiting cleanly. Always either call
        Xvfb.stop() in a finally block, or use Xvfb as a context manager
        to ensure lock files are purged.
        '''
        self._lock_display_file.close()
        try:
            os.remove(self._lock_display_file.name)
        except OSError:
            pass

    def _get_lock_for_display(self, display):
        '''
        In order to ensure multi-process safety, this method attempts
        to acquire an exclusive lock on a temporary file whose name
        contains the display number for Xvfb.
        '''
        tempfile_path = os.path.join(self._tempdir, '.X{0}-lock'.format(display))
        try:
            self._lock_display_file = open(tempfile_path, 'w')
        except PermissionError as e:
            return False
        else:
            try:
                fcntl.flock(self._lock_display_file,
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            else:
                return True

    def _get_next_unused_display(self):
        '''
        Randomly chooses a display number and tries to acquire a lock for this number.
        If the lock could be acquired, returns this number, otherwise choses a new one.
        :return: free display number
        '''
        while True:
            rand = randint(1, self.__class__.MAX_DISPLAY)
            if self._get_lock_for_display(rand):
                return rand
            else:
                continue

    def _set_display_var(self, display):
        os.environ['DISPLAY'] = ':{}'.format(display)



def get_camera_custom(center=(0.5, 0.3, 0.5), distance=3.4, azimuth=-125, elevation=30):
    target = np.array(center)
    theta = 90 + azimuth
    y = distance * math.sin(math.radians(elevation))
    x = math.cos(math.radians(theta)) * distance * math.cos(math.radians(elevation))
    z = math.sin(math.radians(theta)) * distance * math.cos(math.radians(elevation))
    origin = target + np.array([x, y, z])

    return [
        origin.tolist(),
        target.tolist(),
        (0.0, 1.0, 0.0),
    ]
