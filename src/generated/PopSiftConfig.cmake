################################################################################
#
# PopSift - a CUDA implementation of the SIFT algorithm
#
# Copyright 2016, Simula Research Laboratory
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

# Config file for PopSift.
#
# This file is used by CMake when find_package(PopSift) is invoked and either
# the directory containing this file either is present in CMAKE_MODULE_PATH
# (if PopSift was installed), or exists in the local CMake package registry if
# the PopSift build directory was exported.
#
# This module defines a namespace PopSift:: and the target needed to compile and
# link against the library. The target automatically propagate the dependencies
# of the library.
#
# In your CMakeLists.txt  file just add the dependency
#
# find_package(PopSift CONFIG REQUIRED)
#
# Then if you want to link it to an executable
#
# add_executable(poptest yourfile.cpp)
#
# Then to the library
#
# target_link_libraries(poptest PUBLIC PopSift::popsift)
#
# Note that target_include_directories() is not necessary.
#
################################################################################


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
find_dependency(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/PopSiftTargets.cmake")
check_required_components("PopSift")
