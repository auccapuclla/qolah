from datetime import datetime
from datetime import timedelta
from functools import cached_property
import logging
import time
import numpy as np
import os
import re
import struct
from _io import BufferedRandom
import numpy
class VHFparser:
    # Written with reference to SVN r14
    """ 
    Class that parses binary, hexadecimal and ASCII output from VHF board,
    from file objects and data streams.

    Init-time available properties
    -----
    filesize: Number of bytes 
    header: Dictionary containing all paramaters used to call runVHF

    Available methods
    -----
    parse_header: Gets relevant front number of bytes and returns 
    read_words_numpy: Converts binary words into (Q,I,M) arrays

    Run-time available properties
    -----
    reduced_phase: Phase/2pi array
    """

    def __init__(self, filename: str | os.PathLike | BufferedRandom, skip: int):
        """Take a VHF output file and populates relevant properties."""
        self.__create_logger()
        self.logger.info('VHF parser has been run for file: %s', filename)
        # Create class specific modifiers
        self._num_read_bytes = 0
        self._fix_m_called = False

        # Begin Parsing logic
        if isinstance(filename, BufferedRandom):
            self.filesize = filename.tell()
            filename.seek(0)
            self._init_buffer(filename)
        elif isinstance(filename, str) or isinstance(filename, os.PathLike):
            self.filesize = os.path.getsize(filename)  # number of bytes
            self._init_file(filename)
        else:
            self.logger.warning("No file-like/buffer object given to init.")
            print("No file-like/buffer object given to init.")
            return

        # populate body "data"
        max_data_size = int((self.filesize - self._num_read_bytes) / 8)
        # print(max_data_size)
        self.data = np.memmap(
            filename,
            dtype=np.dtype(np.uint64).newbyteorder("<"),
            mode="r",
            offset=self._num_read_bytes,
            shape=(max_data_size,),
        )
        # print(self.data.shape)
        # print(self.data[0])
        # get (I, Q, M)
        self.read_words_numpy(self.data, skip= skip)
        # print(self.q_arr[0:10])

    def __create_logger(self):
        self.logger = logging.getLogger("vhfparser")

    def _init_file(self, filename):
        """For files. Opens file object, and parse by _init_buffer."""
        with open(filename, "rb") as file:  # binary read
            self._init_buffer(file)

    def _init_buffer(self, buffer):
        """For bufferedRandom."""
        header = buffer.read(8)
        if 0xFFFFFFFFFFFF0000 & int.from_bytes(header, "little") != 0x123456ABCDEF0000:
            self.logger.error(
                "Buffer not found to conform to header expectations.")
            self.logger.debug(
                "Buffer read(1024): %s", header + buffer.read(min(1024, self.filesize-8))
            )
            raise ValueError(
                "Buffer does not conform to header expectations.")

        # 1 to account for first word used to determine size of header
        header_count_b = (int.from_bytes(header, "little") & 0xFFFF) - 1
        header_count = header_count_b * 8
        self._num_read_bytes += header_count
        self.headerraw: bytes = buffer.read((header_count)).rstrip(b"\x00")

        self.parse_header(self.headerraw)

    # def read_word(self, b: bytes, idx=int):
    #     """converts 8 bytes in (I, Q, M)."""
    #     m = struct.unpack_from('<h', b, 6)[0]
    #     i = struct.unpack_from('<i', b, 2)[0] >> 8
    #     # i = struct.unpack_from('<i', b, 3)[0] & 0xffffff
    #     q = struct.unpack_from('<i', b, 0)[0] & 0xffffff
    #
    #     sign_extend = -1 & ~0x7fffff
    #     if i & 0x800000:
    #         i |= sign_extend
    #     # i = i - (i >> 23) * 2**24 # branchless
    #     if q & 0x800000:
    #         q |= sign_extend
    #
    #     # substitute for non-local variable
    #     self.m_arr[idx] = m
    #     self.i_arr[idx] = i
    #     self.q_arr[idx] = q

    def read_words_numpy(self, data: np.ndarray, fix_m: bool = False, skip: int = 0):
        """Convert binary words into numpy arrays.

        Input
        -----
        data: np.ndarray
            This contains numpy binary data that contains 1 word of data
            per array element.
        fix_m: bool
            Accounts for 16-bit overflow of m during reading.
        """
        self.i_arr = np.bitwise_and(
            np.right_shift(data, 24), 0xFFFFFF, dtype=np.dtype(np.int32)
        )
        self.i_arr = self.i_arr - (self.i_arr >> 23) * 2**24
        self.q_arr = np.bitwise_and(data, 0xFFFFFF, dtype=np.dtype(np.int32))
        self.q_arr = self.q_arr - (self.q_arr >> 23) * 2**24
        self.m_arr = np.right_shift(data, 48, dtype=np.dtype(np.int64))
        # self.m_arr = self._m_arr_copy.astype(np.int16)
        if skip > 0:
            self.i_arr = self.i_arr[skip:]
            self.q_arr = self.q_arr[skip:]
            self.m_arr = self.m_arr[skip:]
        # failed alternative to m_arr creation
        # self.m_arr.dtype = np.int16 # in-place data change, may be depreciated
        # ? setting the dtype seems to be as buggy as ndarray.view() method
        # this is in contrast to ndarray.astype() method, which returns a copy

        if fix_m:
            self._fix_overflow_m()

    def parse_header(self, header_raw: bytes):
        """Bytes as read from header of bin files is converted into a header property."""
        if header_raw is None:
            raise ValueError("header_raw was not given.")

        # init
        self.header: dict = dict()
        header_raw: list[bytes] = header_raw.split(b"# ")[1:]
        # print(f"Debug: {header_raw = }")
        for x in header_raw[0].split(b" -")[1:]:  # command line record
            x = x.decode().strip(" ")
            self.header[x[0]] = x[1:].strip()
        aux = header_raw[1].split(b': ')[1].split(b'\n')[0].decode().strip()
        aux = datetime.strptime(aux, "%Y-%m-%dT%H:%M:%S%z").isoformat()
        self.header["Time start"] = datetime.fromisoformat(aux)
        for k, v in self.header.items():
            try:
                self.header[k] = int(v)
            except ValueError:
                pass
            except TypeError:
                pass

        # generate explicit params
        if 'l' in self.header:
            self.header["base sampling freq"] = 10e6
        elif 'h' in self.header:
            self.header["base sampling freq"] = 20e6
        else:
            self.logger.warning(
                "Sampling frequency was not explictly given in header. "
                "Defaulting to 20 MHz.")
            self.header["base sampling freq"] = 20e6

        if self.header['s'] is not None:
            self.header["sampling freq"] = self.header["base sampling freq"] / (
                1 + self.header["s"]
            )

    def ignore_first(self, val: int):
        """Drops val number of datapoints from the left.

        Timing information is correspondingly updated.
        """
        try:
            val = int(val)
        except TypeError:
            raise ValueError(f"{val = } given could not be cast to int!")
        if val == 0:
            print("Recieved val = 0 in ignore_first.")
            return
        elif val < 0:
            raise ValueError(f"val given is too small!")
        self.i_arr = self.i_arr[val:]
        self.q_arr = self.q_arr[val:]
        self.m_arr = self.m_arr[val:]
        self.header["Time start"] += timedelta(seconds=(val / self.header["sampling freq"]))

    @cached_property
    def _potential_m_overflow(self) -> bool:
        """Return true if there exists elements in self.m_arr close exceeds +-tol."""
        # Used to calibrate invokation of _m_fix()
        tol: int = 0x7F00
        if tol < 0:
            raise ValueError
        if np.size(np.where(self.m_arr > tol)) + np.size(np.where(self.m_arr < -tol)) > 0:
            return True
        return False

    def _fix_overflow_m(self) -> None:
        """Class method for fixing m value with 16-bit overflow."""
        if not self._fix_m_called:
            # get m_shift of +- n as according before multiplying by 2**16 to
            # account for overflow

            # 1. get location of +/-ve overflows
            # index 0 being +1 would mean that m[0] has flowed above the 16-bit
            # limit into a negative m[1], similarly, index 1 being -1 would
            # mean that m[1] has flowed below the 16-bit minimum into a large
            # positive m[2]
            deltas = np.diff(self.m_arr)  # subtraction between views
            # 2. now round towards +-1
            tol = 0xF000  # keep as int
            deltas = np.greater(deltas, tol).astype(
                int) - np.less(deltas, -tol).astype(int)
            # 3. finally fix
            self.m_arr[1:] -= 0x10000 * np.cumsum(deltas)
            self._fix_m_called = True
        else:
            print("overflow m_arr has been fixed before!")

    @cached_property
    def reduced_phase(self):
        """Obtains phase(time)/2pi for the given file.

        phase is obtained by atan(Q/I).
        """

        # Account for 16 bit overflow of m.
        if not self._fix_m_called and self._potential_m_overflow:
            self._fix_overflow_m()

        phase = -np.arctan2(self.i_arr, self.q_arr)
        phase /= 2*np.pi
        phase -= self.m_arr
        return phase

    @cached_property
    def radii(self):
        """Obtains radius(time) for the given file.

        Radius is obtained by sqrt(Q^2 + I^2).
        """
        radii = np.sqrt(np.power(self.q_arr, 2) + np.power(self.i_arr, 2))
        return radii


if __name__ == "__main__":
    # x = VHFparser(os.path.join(os.path.dirname(__file__), 'vhf_func_gen/Data/60.000_020MHz.txt'))
    # print(f'{x.header = }')
    # print(f"{x.m_arr[-12:] = }")
    filename = "2023-11-20T21-44-55.465328_q268435456_s249_l_b_v3_laser_chip_ULN00238_laser_driver_M00435617_laser_curr_392.8mA_port_number_5.bin"
    start_time = time.time()

    y = VHFparser(os.path.join(os.path.dirname(__file__),
                               filename))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    print()
    # print(f'{y.header = }')
    # print(f'{y.i_arr[:12] = }\n{y.m_arr[:12] = }')
    # print(f"{y.i_arr[-5:] = }\n{y.m_arr[-5:] = }")
    