IF(NOT EXISTS "/home/rahul/Work/LiveGoggle/Code/popsift/install_manifest.txt")
  MESSAGE(FATAL_ERROR "Cannot find install manifest: /home/rahul/Work/LiveGoggle/Code/popsift/install_manifest.txt")
ENDIF(NOT EXISTS "/home/rahul/Work/LiveGoggle/Code/popsift/install_manifest.txt")

FILE(READ "/home/rahul/Work/LiveGoggle/Code/popsift/install_manifest.txt" files)
# STRING(REGEX REPLACE " " ";" files "${files}")
STRING(REGEX REPLACE "\n" ";" file_list "${files}")
FOREACH(file ${file_list})
  MESSAGE(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  IF(EXISTS "$ENV{DESTDIR}${file}")
    EXEC_PROGRAM(
      "/snap/cmake/1035/bin/cmake" ARGS "-E remove $ENV{DESTDIR}${file}"
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    IF(NOT "${rm_retval}" STREQUAL 0)
      MESSAGE(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    ENDIF(NOT "${rm_retval}" STREQUAL 0)
  ELSE(EXISTS "$ENV{DESTDIR}${file}")
    MESSAGE(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  ENDIF(EXISTS "$ENV{DESTDIR}${file}")
ENDFOREACH(file)

