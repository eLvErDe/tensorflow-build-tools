#!/bin/sh


# Function to display usage and quit with 1
usage() {
  echo "This script will create a tree matching NVIDIA"
  echo "layout by symlinking files from Debian's packages"
  echo "Output folder must not exist"
  echo ""
  echo "Usage: ${0} <output_folder>"
  exit 1
}


# Function to log with prefix and level
log() {
  prefix="[create-cuda-tree.sh]"
  lvl="${1}"
  msg="${2}"
  printf "%s %-7s %s\n" "${prefix}" "${lvl}" "${msg}"
}


# Function to display given error message and quit with 2
fatal_error() {
  log ERROR "${1}"
  exit 2
}


# Function to create a relative symlink for a given file with a given root prefix to a given destination
relative_symlink() {

  src=${1}
  src_root=${2}
  dst_root=${3}
 
  if [ -d "${src}" ]; then return; fi

  test -z "${src}" && fatal_error "Incorrect use of relative_symlink function (missing src parameter)"
  test -z "${src_root}" && fatal_error "Incorrect use of relative_symlink function (missing src_root parameter)"
  test -z "${dst_root}" && fatal_error "Incorrect use of relative_symlink function (missing dst_root parameter)"
  echo "${src}" | grep -q "^${src_root}" || fatal_error "Incorrect use of relative_symlink function (dst is not in dst_root)"

  relative_src=`echo ${src} | sed "s|^${src_root}||"`
  relative_src_folder=`dirname "${relative_src}"`
  src_filename=`basename "${relative_src}"`

  if [ "${relative_src_folder}" = "." ]; then
    dst="${dst_root}/${src_filename}"
    mkdir -p "${dst_root}"
  else
    dst="${dst_root}/${relative_src_folder}/${src_filename}"
    mkdir -p "${dst_root}/${relative_src_folder}"
  fi

  log DEBUG "Symlinking ${src} to ${dst}"
  ln -s "${src}" "${dst}"

}


# Sanity checks
out="${1}"
test -z "${out}" && usage
test -e "${out}" && usage
log INFO "Using output folder ${out}"


# Detect CUDA Debian packages
cuda_package="nvidia-cuda-toolkit"
cuda_dev_package="nvidia-cuda-dev"
cuda_required_lib_packages_prefix="libcublas libcudart libcufft libcurand libcusolver"

cuda_package=`dpkg-query -W -f='${Package}' ${cuda_package} 2>/dev/null`
test -z "${cuda_package}" && fatal_error "Unable to detect CUDA package"
cuda_dev_package=`dpkg-query -W -f='${Package}' ${cuda_dev_package} 2>/dev/null`
test -z "${cuda_dev_package}" && fatal_error "Unable to detect CUDA dev package"

cuda_major_version=`dpkg-query -W -f='${Version}' ${cuda_package} 2>/dev/null | cut -d. -f1,2`
test -z "${cuda_major_version}" && fatal_error "Unable to detect CUDA package major version"

for package_prefix in ${cuda_required_lib_packages_prefix}; do
  package="${package_prefix}${cuda_major_version}"
  package_name=`dpkg-query -W -f='${Package}' ${package} 2>/dev/null`
  if [ -z "${cuda_package}" ]; then
    fatal_error "Unable to detect ${package} package"
  else
    log INFO "${package} package detected"
  fi
done

package="libcupti${cuda_major_version}"
libcupti_package=`dpkg-query -W -f='${Package}' ${package} 2>/dev/null`
if [ -z "${libcupti_package}" ]; then 
  fatal_error "Unable to detect ${package} package"
else
  log INFO "${package} package detected"
fi

package="libcupti-dev"
libcupti_dev_package=`dpkg-query -W -f='${Package}' ${package} 2>/dev/null`
if [ -z "${libcupti_dev_package}" ]; then 
  fatal_error "Unable to detect ${package} package"
else
  log INFO "${package} package detected"
fi

cudnn_package=`dpkg-query -W -f='${Package}' libcudnn? 2>/dev/null`
test -z "${cudnn_package}" && fatal_error "Unable to detect cuDNN package name"
cudnn_dev_package=`dpkg-query -W -f='${Package}' ${cudnn_package}-dev 2>/dev/null`
test -z "${cudnn_dev_package}" && fatal_error "Unable to detect cuDNN dev package name"
cudnn_version=`dpkg-query -W -f='${Version}' ${cudnn_package} 2>/dev/null`
test -z "${cudnn_version}" && fatal_error "Unable to detect cuDNN package version"
cudnn_major_version=`echo ${cudnn_version} | cut -d. -f1,2`
test -z "${cudnn_major_version}" && fatal_error "Unable to detect cuDNN package major version"
nvvm_package=`dpkg-query -W -f='${Package}' libnvvm? 2>/dev/null`
test -z "${nvvm_package}" && fatal_error "Unable to detect NVVM package name"

log INFO "CUDA ${cuda_major_version} package detected"
log INFO "cuDNN ${cudnn_major_version} package detected"
log INFO "NVMM package detected"


# Symlink all libraries package content
for package_prefix in ${cuda_required_lib_packages_prefix}; do

  package="${package_prefix}${cuda_major_version}"
  log INFO "Symlink files from package ${package}"

  for file in `dpkg -L "${package}"`; do
    if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
      relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/lib64"
    fi
  done

done


# .so links and headers in nvidia-cuda-dev package
log INFO "Symlink files from package ${cuda_dev_package}"
for file in `dpkg -L "${cuda_dev_package}"`; do

  # Library
  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/lib64"
  # Header
  elif `echo "${file}" | grep -q '^/usr/include/'`; then
    relative_symlink "${file}" "/usr/include/" "${out}/include"
  fi

done


# nvidia-cuda-toolkit package
log INFO "Symlink files from package ${cuda_package}"
for file in `dpkg -L "${cuda_package}"`; do

  # Binary
  if `echo "${file}" | grep -q '^/usr/bin/'`; then
    relative_symlink "${file}" "/usr/bin/" "${out}/bin"
  # NVVM files
  elif `echo "${file}" | grep -q '^/usr/lib/nvidia-cuda-toolkit/libdevice/'`; then
    relative_symlink "${file}" "/usr/lib/nvidia-cuda-toolkit/libdevice/" "${out}/nvvm/libdevice"
  fi

done


# libcupti
log INFO "Symlink files from package ${libcupti_package}"
for file in `dpkg -L "${libcupti_package}"`; do

  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/extras/CUPTI/lib64"
  fi

done


# libcupti-dev
log INFO "Symlink files from package ${libcupti_dev_package}"
for file in `dpkg -L "${libcupti_dev_package}"`; do

  # Library
  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/extras/CUPTI/lib64"
  # Header
  elif `echo "${file}" | grep -q '^/usr/include/'`; then
    relative_symlink "${file}" "/usr/include/" "${out}/extras/CUPTI/include"
  fi

done


# cuDNN
log INFO "Symlink files from package ${cudnn_package}"
for file in `dpkg -L "${cudnn_package}"`; do

  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/lib64"
  fi

done

# For some amazingly wrong reason it's not looking for .so.6 but .so.6.0 uh ?
cudnn_so="${out}/lib64/libcudnn.so.${cudnn_major_version}"
log INFO "Symlinking ${cudnn_so}.* to ${cudnn_so} (Workaround)"
ln -s "${cudnn_so}".* "${cudnn_so}"


# cuDNN dev
log INFO "Symlink files from package ${cudnn_dev_package}"
for file in `dpkg -L "${cudnn_dev_package}"`; do

  # Library
  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/lib64"
  # Header
  elif `echo "${file}" | grep -q '^/usr/include/'`; then
    relative_symlink "${file}" "/usr/include/" "${out}/include"
  fi

done


# NVVM
log INFO "Symlink files from package ${nvvm_package}"
for file in `dpkg -L "${nvvm_package}"`; do

  if `echo "${file}" | grep -q '^/usr/lib/x86_64-linux-gnu/'`; then
    relative_symlink "${file}" "/usr/lib/x86_64-linux-gnu/" "${out}/nvvm"
  fi

done
