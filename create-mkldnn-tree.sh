#!/bin/sh


mode="${2}"


# Function to display usage and quit with 1
usage() {
  echo "This script will create a tree matching Intel MKL DNN"
  echo "layout by symlinking files from Debian's packages"
  echo "Output folder must not exist"
  echo "Mode might be set to debug to show more logs"
  echo ""
  echo "Usage: ${0} <output_folder> [<mode>]"
  exit 1
}


# Function to log with prefix and level
log() {

  prefix="[create-mkldnn-tree.sh]"
  lvl="${1}"
  msg="${2}"

  if [ "${lvl}" = "DEBUG" ]; then
    if [ "${mode}" != "debug" ]; then
      return
    fi
  fi

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


# Detect MKL DNN Debian packages
mkldnn_package=`dpkg-query -W -f='${Package}' libmkldnn?-nonfree-mkl 2>/dev/null`
test -z "${mkldnn_package}" && fatal_error "Unable to detect Intel MKL DNN package name"
mkldnn_dev_package=`dpkg-query -W -f='${Package}' libmkldnn-nonfree-dev 2>/dev/null`
test -z "${mkldnn_dev_package}" && fatal_error "Unable to detect Intel MKL DNN dev package name"
mkldnn_doc_package=`dpkg-query -W -f='${Package}' libmkldnn-docs 2>/dev/null`
test -z "${mkldnn_doc_package}" && fatal_error "Unable to detect Intel MKL DNN doc package name"
log INFO "Intel MKL DNN packages ${mkldnn_package}/${mkldnn_dev_package}/${mkldnn_doc_package} detected"


# Symlink all libraries package content
for package in ${mkldnn_package} ${mkldnn_dev_package} ${mkldnn_doc_package}; do

  log INFO "Symlink files from package ${package}"

  for file in `dpkg -L "${package}"`; do

    # Library
    if `echo "${file}" | grep -q "^/usr/lib/${mkldnn_package}/"`; then
      relative_symlink "${file}" "/usr/lib/${mkldnn_package}/" "${out}/lib"
    elif `echo "${file}" | grep -q '^/usr/lib/'`; then
      relative_symlink "${file}" "/usr/lib/" "${out}/lib"
    # Header
    elif `echo "${file}" | grep -q '^/usr/include/'`; then
      relative_symlink "${file}" "/usr/include/" "${out}/include"
    # Copyright
    elif `echo "${file}" | grep -q "^/usr/share/doc/${mkldnn_doc_package}/copyright\$"`; then
      ln -s "${file}" "${out}/license.txt"
    fi

  done

done
