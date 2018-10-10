#!/usr/bin/python3

# pylint: disable=line-too-long

"""
Run TensorFlow configure script using pexpect module to answer questions
"""

import os
import argparse
from typing import Optional, List
import pexpect


class bool_y_n:
    """
    Boolean value with __str__ returns Y or N instead of True/False
    """

    def __init__(self, value):
        assert value in [True, False], 'bool_y_n __init__ only supports True or False'
        self.value = value

    def __bool__(self):
        return True if self.value else False

    def __repr__(self):
        return 'Y' if self.value else 'N'

    __str__ = __repr__

    def encode(self, encoding):
        """ pexpect wants str like object """
        return self.__str__().encode(encoding)


class PexpectTensorFlowConfigure:
    """
    Run TensorFlow configure script using pexpect module to answer questions
    """

    def __init__(
            self,
            command: str,
            python_version: str,
            with_jemalloc: bool = False,
            with_google_cloud_platform: bool = False,
            with_hdfs: bool = False,
            with_amazon_s3: bool = False,
            with_kafka: bool = False,
            with_xla_jit: bool = False,
            with_gdr: bool = False,
            with_verbs: bool = False,
            with_opencl_sycl: bool = False,
            with_cuda: bool = False,
            with_fresh_clang: bool = False,
            with_mpi: bool = False,
            *,
            cuda_version: Optional[str],
            cuda_path: Optional[str],
            cuda_cudnn_version: Optional[str],
            cuda_cudnn_path: Optional[str],
            cuda_with_tensorrt: bool = False,
            cuda_tensorrt_path: Optional[str],
            cuda_nccl_version: Optional[str],
            cuda_capabilities: Optional[List[str]],
            cuda_with_clang: bool = False,
            cuda_gcc_path: Optional[str],
            mpi_path: Optional[str],
            opts_flags: Optional[List[str]],
        ) -> None:

        assert os.path.exists(command), 'command must be a valid path to TensorFlow configure script'

        if with_opencl_sycl:
            raise NotImplementedError('with_opencl_sycl=True is not implemented')

        if cuda_with_clang:
            raise NotImplementedError('cuda_with_clang=True is not implemented')

        if with_cuda:
            assert cuda_version is not None, 'cuda_version must be set if with_cuda=True'
            assert cuda_path is not None, 'cuda_path must be set if with_cuda=True'
            assert cuda_cudnn_version is not None, 'cuda_cudnn_version must be set if with_cuda=True'
            assert cuda_cudnn_path is not None, 'cuda_cudnn_path must be set if with_cuda=True'
            assert cuda_nccl_version is not None, 'cuda_nccl_version must be set if with_cuda=True'
            assert cuda_capabilities is not None, 'cuda_capabilities must be set if with_cuda=True'
            assert cuda_gcc_path is not None, 'cuda_gcc_path must be set if with_cuda=True'

            assert os.path.exists(cuda_path), 'cuda_path is set a non existing folder'
            assert os.path.exists(cuda_cudnn_path), 'cuda_cudnn_path is set a non existing folder'

            if not cuda_with_clang:
                assert cuda_gcc_path is not None, 'cuda_gcc_path must be set if cuda_with_clang=True'
                assert os.path.exists(cuda_gcc_path), 'cuda_gcc_path is set a non existing file'

            if cuda_with_tensorrt:
                assert cuda_tensorrt_path is not None, 'cuda_tensorrt_path must be set if cuda_with_tensorrt=True'
                assert os.path.exists(cuda_tensorrt_path), 'cuda_tensorrt_path is set a non existing folder'

        if with_mpi:
            assert mpi_path is not None, 'mpi_path must be set if with_mpi=True'
            assert os.path.exists(mpi_path), 'mpi_path is set a non existing folder'

        self.command = command
        self.python_version = python_version
        self.with_jemalloc = bool_y_n(with_jemalloc)
        self.with_google_cloud_platform = bool_y_n(with_google_cloud_platform)
        self.with_hdfs = bool_y_n(with_hdfs)
        self.with_amazon_s3 = bool_y_n(with_amazon_s3)
        self.with_kafka = bool_y_n(with_kafka)
        self.with_xla_jit = bool_y_n(with_xla_jit)
        self.with_gdr = bool_y_n(with_gdr)
        self.with_verbs = bool_y_n(with_verbs)
        self.with_opencl_sycl = bool_y_n(with_opencl_sycl)
        self.with_cuda = bool_y_n(with_cuda)
        self.with_fresh_clang = bool_y_n(with_fresh_clang)
        self.with_mpi = bool_y_n(with_mpi)
        self.cuda_version = cuda_version
        self.cuda_path = cuda_path
        self.cuda_cudnn_version = cuda_cudnn_version
        self.cuda_cudnn_path = cuda_cudnn_path
        self.cuda_with_tensorrt = bool_y_n(cuda_with_tensorrt)
        self.cuda_tensorrt_path = cuda_tensorrt_path
        self.cuda_nccl_version = cuda_nccl_version
        self.cuda_capabilities = cuda_capabilities
        self.cuda_with_clang = bool_y_n(cuda_with_clang)
        self.cuda_gcc_path = cuda_gcc_path
        self.mpi_path = mpi_path
        self.opts_flags = opts_flags

        # Pexpect script will be stored here
        self.script = None


    @property
    def python_bin_path(self):
        """ Returns /usr/bin/python$VER """

        python_bin_path = '/usr/bin/python%s' % self.python_version
        assert os.path.exists(python_bin_path), '%s does not exist' % python_bin_path
        return python_bin_path


    @property
    def python_dist_path(self):
        """ Returns /usr/lib/python3/dist-packages or /usr/lib/python$VER/dist-packages (Python 2) """

        if self.python_version.startswith('2.'):
            python_dist_path = '/usr/lib/python%s/dist-packages' % self.python_version
        else:
            python_dist_path = '/usr/lib/python3/dist-packages'
        assert os.path.exists(python_dist_path), '%s does not exist' % python_dist_path
        return python_dist_path


    @property
    def opts_flags_str(self):
        """ Return opts flags space separated or empty string is None """

        return '' if self.opts_flags is None else ' '.join([x.strip() for x in self.opts_flags])


    @property
    def cuda_capabilities_str(self):
        """ Return coma separated CUDA capabilities """

        return ','.join([x.strip() for x in self.cuda_capabilities])


    @staticmethod
    def command_line_args():
        """ Create a command line arguments to run this as a cli script """

        parser = argparse.ArgumentParser(description='Run TensorFlow configure script non interactively', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('command', type=str, help='TensorFlow configure script to run')
        parser.add_argument('python_version', type=str, help='Major Python version to build for')

        parser.add_argument('--with-jemalloc', action='store_true', help='Use jemalloc as malloc')
        parser.add_argument('--with-google-cloud-platform', action='store_true', help='Enable Google Cloud Platform support')
        parser.add_argument('--with-hdfs', action='store_true', help='Enable Hadoop File System support')
        parser.add_argument('--with-amazon-s3', action='store_true', help='Enable Amazon S3 File System support')
        parser.add_argument('--with-kafka', action='store_true', help='Enable Apache Kafka Platform support')
        parser.add_argument('--with-xla-jit', action='store_true', help='Enable XLA JIT support')
        parser.add_argument('--with-gdr', action='store_true', help='Enable GDR support')
        parser.add_argument('--with-opencl-sycl', action='store_true', help='Enable OpenCL SYCL support')
        parser.add_argument('--with-cuda', action='store_true', help='Enable NVIDIA CUDA support')
        parser.add_argument('--with-fresh-clang', action='store_true', help='Download a fresh release of clang')
        parser.add_argument('--with-mpi', action='store_true', help='Enable MPI support')
 
        parser.add_argument('--cuda-version', type=str, help='Major NVIDIA CUDA version', metavar='9.0')
        parser.add_argument('--cuda-path', type=str, help='Path to CUDA installation folder', metavar='/usr/local/cuda')
        parser.add_argument('--cuda-cudnn-version', type=str, help='Major cuDNN version', metavar='7.0')
        parser.add_argument('--cuda-cudnn-path', type=str, help='Path to cuDNNN installation folder', metavar='/usr/local/cuda')
        parser.add_argument('--cuda-with-tensorrt', action='store_true', help='Enable NVIDIA TensorRT for inference')
        parser.add_argument('--cuda-tensorrt-path', type=str, help='Path to TensorRT installation folder', metavar='/usr/lib/x86_64-linux-gnu')
        parser.add_argument('--cuda-nccl-version', type=str, help='Major NVIDIA NCCL version', metavar='1.3')
        parser.add_argument('--cuda-capabilities', type=str, nargs='+', help='CUDA capabilities to enable', metavar=('3.5', '5.2'))
        parser.add_argument('--cuda-with-clang', action='store_true', help='Use CLANG as CUDA compiler')
        parser.add_argument('--cuda-gcc-path', type=str, help='Path to GCC to use for CUDA compilation', metavar='/usr/bin/gcc')

        parser.add_argument('--mpi-path', type=str, help='Path to (Open)MPI installation folder', metavar='/usr')

        parser.add_argument('--opts-flags', type=str, nargs='+', help='Optimization flags to use during compilation. QUOTE VALUES. ADD TRAILING OR LEADING SPACE IF IT IS A SINGLE VALUE', metavar=('" -march=native"', '" -mavx2"'))

        parsed = parser.parse_args()
        return parsed


    def _pexpect_line_answer(self, question, answer):
        """ Expect a pattern an provide answer (also display output) """

        if self.script.after is not None:
            print(self.script.after.decode('utf-8'))
        self.script.expect(question)
        if answer is not None:
            self.script.sendline(answer)
        if self.script.before is not None:
            print(self.script.before.decode('utf-8'))


    def run(self):
        """ Run expect against TF configure script an provide answers """

        self.script = pexpect.spawn(self.command)
        self.script.timeout = 2

        self._pexpect_line_answer(r'Please specify the location of python\..*', self.python_bin_path)
        self._pexpect_line_answer(r'Please input the desired Python library path to use\..*', self.python_dist_path)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with jemalloc as malloc support?.*', self.with_jemalloc)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with Google Cloud Platform support?.*', self.with_google_cloud_platform)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with Hadoop File System support?.*', self.with_hdfs)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with Amazon S3 File System support?.*', self.with_amazon_s3)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with Apache Kafka Platform support?.*', self.with_kafka)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with XLA JIT support?.*', self.with_xla_jit)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with GDR support?.*', self.with_gdr)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with VERBS support?.*', self.with_verbs)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with OpenCL SYCL support?.*', self.with_opencl_sycl)
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with CUDA support?.*', self.with_cuda)

        # CUDA
        if self.with_cuda:
            self._pexpect_line_answer(r'Please specify the CUDA SDK version you want to use.*', self.cuda_version)
            self._pexpect_line_answer(r'Please specify the location where CUDA .* toolkit is installed\..*', self.cuda_path)
            self._pexpect_line_answer(r'Please specify the cuDNN version you want to use\..*', self.cuda_cudnn_version)
            self._pexpect_line_answer(r'Please specify the location where cuDNN .* library is installed\..*', self.cuda_cudnn_path)
            self._pexpect_line_answer(r'Do you wish to build TensorFlow with TensorRT support?.*', self.cuda_with_tensorrt)

            # CUDA TensorRT
            if self.cuda_with_tensorrt:
                self._pexpect_line_answer(r'Please specify the location where TensorRT is installed\..*', self.cuda_tensorrt_path)

            self._pexpect_line_answer(r'Please specify the NCCL version you want to use\..*', self.cuda_nccl_version)
            self._pexpect_line_answer(r'Please specify a list of comma-separated Cuda compute capabilities you want to build with\..*', self.cuda_capabilities_str)

            self._pexpect_line_answer(r'Do you want to use clang as CUDA compiler?.*', self.cuda_with_clang)

            # CUDA use Clang
            if self.cuda_with_clang:
                raise NotImplementedError('cuda_with_clang=True is not implemented')
            else:
                self._pexpect_line_answer(r'Please specify which gcc should be used by nvcc as the host compiler\..*', self.cuda_gcc_path)

        else:
            # Is not prompted in CUDA mode
            self._pexpect_line_answer(r'Do you wish to download a fresh release of clang?.*', self.with_fresh_clang)

        # MPI
        self._pexpect_line_answer(r'Do you wish to build TensorFlow with MPI support?.*', self.with_mpi)
        if self.with_mpi:
            self._pexpect_line_answer(r'Please specify the MPI toolkit folder\..*', self.mpi_path)

        self._pexpect_line_answer(r'Please specify optimization flags to use during compilation when bazel option.*', self.opts_flags_str)
        self._pexpect_line_answer(r'Would you like to interactively configure \./WORKSPACE for Android builds?.*', 'N')
        self._pexpect_line_answer(r'Configuration finished', None)
        self._pexpect_line_answer(pexpect.EOF, None)


if __name__ == '__main__':


    CONFIG = PexpectTensorFlowConfigure.command_line_args()

    EXPECT = PexpectTensorFlowConfigure(**vars(CONFIG))

    EXPECT.run()
