# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

include(CMakeDependentOption)

# megamol_plugin()
#
# Parameters:
#   PLUGIN_NAME:       name of the plugin, should match the plugin directory name.
#   DIR:               full path to the plugin directory, defaults to CMAKE_CURRENT_SOURCE_DIR if not set
#   BUILD_DEFAULT:     ON/OFF if the plugin should be build with the default configuration
#   DEPENDS_PLUGINS:   list of plugins which are required by this plugin
#   DEPENDS_EXTERNALS: list of externals which are required by this plugin

function(megamol_plugin PLUGIN_NAME)
  # Parse arguments
  set(optionArgs ALWAYS_BUILD DEPENDS_CUDA DEPENDS_GL)
  set(oneValueArgs DIR BUILD_DEFAULT)
  set(multiValueArgs DEPENDS_PLUGINS DEPENDS_EXTERNALS)
  cmake_parse_arguments(MMPLUGIN_ARGS "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Validate plugin name
  if (NOT "${PLUGIN_NAME}" MATCHES "^[A-Za-z0-9_]+$")
    message(FATAL_ERROR "Plugin name is only allowed to contain letters, numbers and underscore, found: ${PLUGIN_NAME}.")
  endif ()

  # Upper case plugin name
  string(TOUPPER ${PLUGIN_NAME} PLUGIN_NAME_UPPER)

  # Check feature requirements
  set(REQUIREMENTS_AVAILABLE TRUE)

  # Check Cuda
  if (MMPLUGIN_ARGS_DEPENDS_CUDA)
    if (NOT ENABLE_CUDA)
      set(REQUIREMENTS_AVAILABLE FALSE)
    else ()
      enable_language(CUDA)
    endif ()
  endif ()

  # Check GL
  if (MMPLUGIN_ARGS_DEPENDS_GL)
    if (NOT ENABLE_GL)
      set(REQUIREMENTS_AVAILABLE FALSE)
    endif ()
  endif ()

  # Enable plugin build option, show only if requirements are available and not always build
  cmake_dependent_option(BUILD_PLUGIN_${PLUGIN_NAME_UPPER} "Option to build ${PLUGIN_NAME}" "${MMPLUGIN_ARGS_BUILD_DEFAULT}" "REQUIREMENTS_AVAILABLE;NOT MMPLUGIN_ARGS_ALWAYS_BUILD" OFF)

  if ((BUILD_PLUGIN_${PLUGIN_NAME_UPPER} OR MMPLUGIN_ARGS_ALWAYS_BUILD) AND REQUIREMENTS_AVAILABLE)

    # Set plugin enabled flag
    set(${PLUGIN_NAME}_PLUGIN_ENABLED ON PARENT_SCOPE)

    # Default DIR argument to CMAKE_CURRENT_SOURCE_DIR
    if (NOT DEFINED MMPLUGIN_ARGS_DIR)
      set(MMPLUGIN_ARGS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    endif ()

    # Common dirs
    set(MMPLUGIN_DIR_INCLUDE "${MMPLUGIN_ARGS_DIR}/include")
    set(MMPLUGIN_DIR_SRC "${MMPLUGIN_ARGS_DIR}/src")

    # Validate include dir (if present) only contains a single dir "<plugin-name>".
    if (IS_DIRECTORY "${MMPLUGIN_DIR_INCLUDE}")
      file(GLOB include_subdirs RELATIVE "${MMPLUGIN_DIR_INCLUDE}" "${MMPLUGIN_DIR_INCLUDE}/*")
      list(LENGTH include_subdirs list_size)
      if ((NOT list_size EQUAL 1) OR (NOT PLUGIN_NAME IN_LIST include_subdirs) OR (NOT IS_DIRECTORY "${MMPLUGIN_DIR_INCLUDE}/${PLUGIN_NAME}"))
        message(FATAL_ERROR "\"${MMPLUGIN_DIR_INCLUDE}\" must contain exactly one subdirectory \"${PLUGIN_NAME}\".")
      endif ()
    endif ()

    # Collect files within default plugin structure
    file(GLOB_RECURSE public_header_files RELATIVE "${MMPLUGIN_ARGS_DIR}" "${MMPLUGIN_DIR_INCLUDE}/${PLUGIN_NAME}/*.h")
    file(GLOB_RECURSE header_files RELATIVE "${MMPLUGIN_ARGS_DIR}" "${MMPLUGIN_DIR_SRC}/*.h")
    file(GLOB_RECURSE source_files RELATIVE "${MMPLUGIN_ARGS_DIR}" "${MMPLUGIN_DIR_SRC}/*.cpp")
    file(GLOB_RECURSE shader_files RELATIVE "${MMPLUGIN_ARGS_DIR}" "${MMPLUGIN_ARGS_DIR}/shaders/*")

    # Target definition
    add_library(${PLUGIN_NAME} OBJECT ${public_header_files} ${header_files} ${source_files} ${shader_files})
    target_include_directories(${PLUGIN_NAME}
      PUBLIC $<BUILD_INTERFACE:${MMPLUGIN_DIR_INCLUDE}>
      PRIVATE $<BUILD_INTERFACE:${MMPLUGIN_DIR_SRC}>)

    # List of reserved words in plugins or externals dependencies lists.
    # This keywords are passed to target_link_libraries, but are ignored on other processing.
    set(reserved_link_keywords "PUBLIC" "PRIVATE" "INTERFACE")

    # Save all required plugins in global list. Required plugin may not be defined, yet.
    # Therefore, delay check until all plugins are added.
    foreach (required_plugin ${MMPLUGIN_ARGS_DEPENDS_PLUGINS})
      if (NOT "${required_plugin}" IN_LIST reserved_link_keywords)
        list(APPEND required_plugins_list "${PLUGIN_NAME} ${required_plugin}")
        set(required_plugins_list ${required_plugins_list} CACHE INTERNAL "")
      endif ()
    endforeach ()

    # Externals
    foreach (external ${MMPLUGIN_ARGS_DEPENDS_EXTERNALS})
      if (NOT "${external}" IN_LIST reserved_link_keywords)
        require_external(${external})
      endif ()
    endforeach ()

    # Link
    target_link_libraries(${PLUGIN_NAME}
      PUBLIC OpenMP::OpenMP_CXX
      PRIVATE core)
    if (ENABLE_GL)
      target_link_libraries(${PLUGIN_NAME} PRIVATE core_gl)
    endif ()
    target_link_libraries(${PLUGIN_NAME}
      PRIVATE ${MMPLUGIN_ARGS_DEPENDS_PLUGINS}
      PRIVATE ${MMPLUGIN_ARGS_DEPENDS_EXTERNALS})

    # Install shaders
    if (EXISTS "${MMPLUGIN_ARGS_DIR}/shaders")
      install(DIRECTORY ${MMPLUGIN_ARGS_DIR}/shaders/ DESTINATION "share/shaders")
    endif ()
    # Install resources
    if (EXISTS "${MMPLUGIN_ARGS_DIR}/resources")
      install(DIRECTORY ${MMPLUGIN_ARGS_DIR}/resources/ DESTINATION "share/resources")
    endif ()

    # Grouping in Visual Studio
    set_target_properties(${PLUGIN_NAME} PROPERTIES
      FOLDER plugins)
    source_group(TREE "${MMPLUGIN_DIR_INCLUDE}/${PLUGIN_NAME}" PREFIX "Public Header Files" FILES ${public_header_files})
    source_group(TREE "${MMPLUGIN_DIR_SRC}" PREFIX "Header Files" FILES ${header_files})
    source_group(TREE "${MMPLUGIN_DIR_SRC}" PREFIX "Source Files" FILES ${source_files})
    source_group(TREE "${MMPLUGIN_ARGS_DIR}/shaders" PREFIX "Shader Files" FILES ${shader_files})

    # Register plugin
    target_link_libraries(plugins INTERFACE ${PLUGIN_NAME})
    target_sources(plugins INTERFACE $<TARGET_OBJECTS:${PLUGIN_NAME}>)
  endif ()
endfunction()

# Create plugin target
#
# We need an extra target for all plugins to avoid cyclic dependencies, which are only allowed for static libraries,
# see https://gitlab.kitware.com/cmake/cmake/-/issues/17905.
# Each plugin must link the core to have access to the plugin registry and probably common stuff, like calls, ...
# Therefore, we cannot just link from the core back to all plugins, to have a single target, which can be used by the
# frontend(s). Maybe if CMake supports this in future we can change to this setup.
# Instead, for now, define an extra 'plugins' target. This target will contain all plugin objects. Frontends can just
# link 'core' and 'plugins' targets and have not to deal with individual plugins.
add_library(plugins INTERFACE)

# Global list of required plugins
set(required_plugins_list "" CACHE INTERNAL "")

# Find and add plugins
file(GLOB plugins RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" *)
foreach (plugin ${plugins})
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt" AND NOT ${plugin} STREQUAL "doc_template")
    add_subdirectory(${plugin})
  endif ()
endforeach ()

# Validate if all required plugins are added now.
foreach (plugin_dep ${required_plugins_list})
  # Split plugin_dep. It is formatted "plugin_name dependency_name".
  string(REGEX REPLACE " +" ";" plugin_dep "${plugin_dep}")
  list(GET plugin_dep 0 plugin_name)
  list(REMOVE_AT plugin_dep 0)
  if (NOT TARGET ${plugin_dep})
    message(FATAL_ERROR "Plugin \"${plugin_name}\" requires \"${plugin_dep}\", but it is not enabled!")
  endif ()
endforeach ()
