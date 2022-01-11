#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PopSift::popsift" for configuration "Release"
set_property(TARGET PopSift::popsift APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(PopSift::popsift PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpopsift.so.1.0.0"
  IMPORTED_SONAME_RELEASE "libpopsift.so.1.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS PopSift::popsift )
list(APPEND _IMPORT_CHECK_FILES_FOR_PopSift::popsift "${_IMPORT_PREFIX}/lib/libpopsift.so.1.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
