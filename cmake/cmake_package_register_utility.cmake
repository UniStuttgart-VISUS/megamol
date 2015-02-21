#
# VISlib - package_register_utilty
# Copyright (C) 2015 by Sebastian Grottel
# Alle Rechte vorbehalten. All rights reserved.
#
cmake_minimum_required(VERSION 2.8)

#
# This function prepares a 'path' file to be installed into the cmake package user registry.
# The file is generated inside ${CMAKE_CURRENT_BINARY_DIR}.
#
# This function ONLY works on Unix os.
#
# @param package_name              STRING  The name of the package to be installed
# @param package_install_dir       STRING  The path the package will be installed to, i.e. the path where the file "<package_name>Config.cmake" can be found
# @param out_package_registry_file VAR     Name of the variable which will receive the file name of the generated path file
# @param out_package_user_registry VAR     Name of the variable which will receive the path of the cmake package user registry. This path includes the subdirectory for the package itself
#
# Example:
#   set(package_name "vislib")
#   set(package_install_dir "${CMAKE_INSTALL_PREFIX}/share/cmake/${package_name}")
#   vislibutil_prepare_cmake_package_user_repository("${package_name}" "${package_install_dir}" package_registry_file package_user_registry)
#
# Variables have been set to:
#   ${package_registry_file} to "/home/username/vislib/build.release/b7ae7358525651e84fd5b0812d7ffaf9"
#   ${package_user_registry} to "/home/username/.cmake/packages/vislib"
#
function(vislibutil_prepare_cmake_package_user_repository
		package_name
		package_install_dir
		out_package_registry_file
		out_package_user_registry)
	# some sanity checks
	if (NOT ${UNIX})
		message(FATAL_ERROR "function vislibutil_prepare_cmake_package_user_repository is only supported on Unix-like operating systems")
	endif()

	# some output to be sure the parameters are set correctly
#	message(STATUS "package_name == ${package_name}")
#	message(STATUS "package_install_dir == ${package_install_dir}")

	# create a temp file in the built tree holding the installation path
	set(tmp_file_name "${CMAKE_CURRENT_BINARY_DIR}/inst_tmp")
	file(WRITE ${tmp_file_name} "${package_install_dir}")
	# compute a content hash to be the files final name
	file(MD5 ${tmp_file_name} package_registry_hash)
	set(reg_file_name "${CMAKE_CURRENT_BINARY_DIR}/${package_registry_hash}")
	# rename the file
	file(RENAME ${tmp_file_name} ${reg_file_name})
	# paranoia: just be sure to remove the temp file
	file(REMOVE ${tmp_file_name})

	# return the results variable
	set(${out_package_registry_file} ${reg_file_name} PARENT_SCOPE)
	set(${out_package_user_registry} "$ENV{HOME}/.cmake/packages/${package_name}" PARENT_SCOPE)

endfunction(vislibutil_prepare_cmake_package_user_repository)

