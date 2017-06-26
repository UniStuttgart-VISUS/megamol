#!/bin/bash
#
# MegaMol mdao2
# building utilty script
# Copyright 2015 by MegaMol Team
# All rights reserved
#
#  use '-h' for help
#
# default behavior:
#   - create build tree subdirectory
#   - invokes cmake without special arguments
#   - invokes make without special arguments
#

install_prefix=
build_debug=0
build_release=1
invoke_cmake=1
invoke_make=1
invoke_make_install=0
invoke_default=1
cmake_extra_cmd=
MegaMolCore_DIR=
use_mmcore_install_prefix=1
register_build_trees=0


# be proud of yourself
echo
echo "  MegaMol(TM) mdao2"
echo "  cmake_build.sh utilty script"
echo "  Copyright 2015, by MegaMol Team"
echo "  All rights reserved"
echo 

#parse user commands
while getopts "hp:dDcC:mirf:" opt; do
  case $opt in
  h)
    echo "Available command line options:"
    echo "  -h    (help) prints this help message"
    echo "  -p XX (prefix) specifies the installation prefix to be used"
    echo "  -d    (debug) also builds debug version in separate build tree subdirectory"
    echo "  -D    (Debug only) builds only the debug version, not the release version"
    echo "  -c    (cmake) invokes 'cmake'. This also deletes all files and subdirectories which might be present in the build tree subdirectory"
    echo "  -C XX (cmake option) additional command to be passed to 'cmake'"
    echo "  -m    (make) invokes 'make'"
    echo "  -i    (install) invokes 'make install'"
    echo "  -r    (register) registers the cmake build tree results in the user package repository, making them available for find_package commands"
    echo "  -f XX (find hint) specifies the find hint path where the MegaMolCore is located"
    echo
    echo "Default behavior (when no arguments are given):"
    echo "  - creates build tree subdirector 'build.release'"
    echo "  - invokes 'cmake' without special arguments"
    echo "  - invokes 'make' without special arguments"
    echo "This is similar to calling '$0 -cm'"
    echo
    echo "Note: if '-c', '-m', or '-i' is specified, the default behavior is overwritten be the requested invokes."
    echo
    echo "Examples:"
    echo
    echo "$0 -Dc"
    echo
    echo "  creates the build tree subdirectory for the debug version, not for the release version, and invokes 'cmake', but does not invoke 'make' or 'make install'-"
    echo
    echo "$0 -dcmi -p ~/my_inst -f ~/mmcore_inst"
    echo
    echo "  creates the build tree subdirectors for release and debug, and invokes 'cmake', 'make' and 'make install' in both build trees. Additionally the installation prefix is set to '~/my_inst' and as search hint for the MegaMolCore package '~/mmcore_inst' is passed to cmake."
    echo
    exit 0
    ;;
  p)
    install_prefix=$OPTARG
    use_mmcore_install_prefix=0
    ;;
  d)
    build_debug=1
    ;;
  D)
    build_debug=1
    build_release=0
    ;;
  c)
    if [ $invoke_default -eq 1 ] ; then invoke_default=0; invoke_cmake=0; invoke_make=0; invoke_make_install=0; fi
    invoke_cmake=1
    ;;
  C)
    cmake_extra_cmd="$cmake_extra_cmd $OPTARG"
    ;;
  m)
    if [ $invoke_default -eq 1 ] ; then invoke_default=0; invoke_cmake=0; invoke_make=0; invoke_make_install=0; fi
    invoke_make=1
    ;;
  i)
    if [ $invoke_default -eq 1 ] ; then invoke_default=0; invoke_cmake=0; invoke_make=0; invoke_make_install=0; fi
    invoke_make_install=1
    ;;
  r)
    register_build_trees=1
    ;;
  f)
    MegaMolCore_DIR=$OPTARG
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1
    ;;
  esac
done

# prepare command line for cmake
cmake_cmd=""
if [ $install_prefix ] ; then cmake_cmd="$cmake_cmd -DCMAKE_INSTALL_PREFIX=$install_prefix"; fi
if [ $MegaMolCore_DIR ] ; then cmake_cmd="$cmake_cmd -DMegaMolCore_DIR=$MegaMolCore_DIR"; fi
if [ $use_mmcore_install_prefix ] ; then cmake_cmd="$cmake_cmd -DUSE_MEGAMOLCORE_INSTALL_PREFIX=1"; fi
if [ $register_build_trees -eq 1 ] ; then cmake_cmd="$cmake_cmd -Dregister_build_trees=1"; fi
cmake_cmd="$cmake_cmd $cmake_extra_cmd"

# debug output of settings
#echo "Specified settings:"
#echo "  install_prefix=$install_prefix"
#echo "  build_debug=$build_debug"
#echo "  build_release=$build_release"
#echo "  invoke_cmake=$invoke_cmake"
#echo "  invoke_make=$invoke_make"
#echo "  invoke_make_install=$invoke_make_install"
#echo "  invoke_default=$invoke_default"
#echo "  MegaMolCore_DIR=$MegaMolCore_DIR"
#echo "  cmake_cmd=$cmake_cmd"

if [ $build_release -eq 1 ] ; then
  echo "Building release"
  build_dir="build.release"
  if [ $invoke_cmake -eq 1 ] ; then
    echo "create build tree subdirectory"
    rm -rf $build_dir
    mkdir $build_dir
    echo "invoke 'cmake' (-DCMAKE_BUILD_TYPE=Release $cmake_cmd)"
    cd $build_dir
    cmake .. -DCMAKE_BUILD_TYPE=Release $cmake_cmd
    cd ..
  fi
  if [ $invoke_make -eq 1 ] ; then
    echo "invoke 'make'"
    cd $build_dir
    make
    cd ..
  fi
  if [ $invoke_make_install -eq 1 ] ; then
    echo "invoke 'make install'"
    cd $build_dir
    make install
    cd ..
  fi
fi

if [ $build_debug -eq 1 ] ; then
  echo "Building debug"
  build_dir="build.debug"
  if [ $invoke_cmake -eq 1 ] ; then
    echo "create build tree subdirectory"
    rm -rf $build_dir
    mkdir $build_dir
    echo "invoke 'cmake' (-DCMAKE_BUILD_TYPE=Debug $cmake_cmd)"
    cd $build_dir
    cmake .. -DCMAKE_BUILD_TYPE=Debug $cmake_cmd
    cd ..
  fi
  if [ $invoke_make -eq 1 ] ; then
    echo "invoke 'make'"
    cd $build_dir
    make
    cd ..
  fi
  if [ $invoke_make_install -eq 1 ] ; then
    echo "invoke 'make install'"
    cd $build_dir
    make install
    cd ..
  fi
fi

echo "$0 completed"
#if [ $? -ne 0 ] ; then exit 1; fi

