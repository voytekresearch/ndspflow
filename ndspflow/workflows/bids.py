"""BIDS input."""

import os
import warnings
import numpy as np

from mne_bids import BIDSPath, read_raw_bids


class BIDS:
    """BIDS interface.

    Attributes
    ----------
    bids_path : str, optional, default: None
        Path to BIDS directory.
    subjects : list of str, optional, default: None
        Subset of subjects to include.
    fs : float, optional, default: None
        Sampling rate, in Hertz.
    y_array : ndarray
        Array input.
    """
    def __init__(self, bids_path=None, subjects=None, fs=None, **bids_kwargs):
        """Initalize BIDS object.

        Parameters
        ----------
        bids_path : str, optional, default: None
            Path to BIDS directory.
        subjects : list of str, optional, default: None
            Subset of subjects to include.
        fs : float, optional, default: None
            Sampling rate, in Hertz.
        **bids_kwargs
            Additional keyword arguments to pass to mne_bids.path.BIDSPath
            initalization. Examples include: session, task, acquisition, run, etc.
        """

        # Path to bids
        self.bids_path = bids_path

        # Allows subject sub-selection
        if subjects is None and bids_path is not None:
            self.subjects = sorted([sub.strip('sub-') for sub in os.listdir(bids_path)
                                    if 'sub-' in sub])
        elif subjects is not None:
            self.subjects = [sub.strip('sub-') for sub in subjects]
        else:
            self.subjects = subjects

        # MNE BIDSPath initalization kwargs
        self.bids_kwargs = bids_kwargs

        # Ensure unpackable
        self.bids_kwargs = {} if self.bids_kwargs is None else self.bids_kwargs

        # Sampling rate
        self.fs = fs

        # Channel names
        self.ch_names = None

        # Output array
        self.y_array = None

        self.nodes = []


    def read_bids(self, subject=None, allow_ragged=False, queue=True):
        """Read the BIDS directory into memory.

        Parameters
        ----------
        subject : int, optional, default: None
            Read a single subject into memory. If None, the entire BIDS dataset is read
            into memory at once.
        allow_ragged : bool, optional, default: True
            Allow and use ragged arrays if True. Otherwise assumes non-ragged and sets max
            output length to min raw length. Only used if ind is None.
        queue : bool, optional, default: True
            Queue's reading into nodes if True. Otherwise reads y_array in.
        """

        if queue:
            # Queue for later execution
            self.nodes.append(['read_bids', allow_ragged, False])

        elif subject is not None:
            # Read single subject
            subject = subject.strip('sub-')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bids_path = BIDSPath(root=self.bids_path, subject=subject,
                                    **self.bids_kwargs)

                raw = read_raw_bids(bids_path, verbose=False)

            if self.fs is None:
                self.fs = int(raw.info['sfreq'])

            self.y_array = raw.get_data()
            self.ch_names = raw.ch_names

            del raw

        else:
            # Read all subjects
            for ind, sub in enumerate(self.subjects):

                # Raw bids
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    bids_path = BIDSPath(root=self.bids_path, subject=sub, **self.bids_kwargs)
                    raw = read_raw_bids(bids_path, verbose=False)

                # Sampling rate
                fs = int(raw.info['sfreq'])
                if self.fs is None:
                    self.fs = fs
                elif self.fs != fs:
                    raise ValueError('Resample subject data to the same sampling rate.')

                # Channel names
                self.ch_names = raw.ch_names

                # Get array
                arr = raw.get_data()
                del raw

                # Initalize array
                if ind == 0 and not allow_ragged:
                    self.y_array = np.zeros((len(self.subjects), *arr.shape))
                elif ind == 0 and allow_ragged:
                    self.y_array = []

                if allow_ragged:
                    self.y_array.append(arr)
                else:
                    # Trim array if needed
                    y_len = len(self.y_array[ind])

                    if len(arr) < y_len:
                        self.y_array = self.y_array[:, :len(arr)]

                    elif len(arr) > y_len:
                        arr = arr[:y_len]

                    self.y_array[ind] = arr

            if allow_ragged:
                self.y_array = np.array(self.y_array, dtype=object)
