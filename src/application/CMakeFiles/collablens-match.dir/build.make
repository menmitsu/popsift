# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1035/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1035/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rahul/Work/LiveGoggle/Code/popsift

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rahul/Work/LiveGoggle/Code/popsift

# Include any dependencies generated for this target.
include src/application/CMakeFiles/collablens-match.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/application/CMakeFiles/collablens-match.dir/compiler_depend.make

# Include the progress variables for this target.
include src/application/CMakeFiles/collablens-match.dir/progress.make

# Include the compile flags for this target's objects.
include src/application/CMakeFiles/collablens-match.dir/flags.make

src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o: src/application/CMakeFiles/collablens-match.dir/flags.make
src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o: src/application/match_opencv.cpp
src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o: src/application/CMakeFiles/collablens-match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rahul/Work/LiveGoggle/Code/popsift/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o -MF CMakeFiles/collablens-match.dir/match_opencv.cpp.o.d -o CMakeFiles/collablens-match.dir/match_opencv.cpp.o -c /home/rahul/Work/LiveGoggle/Code/popsift/src/application/match_opencv.cpp

src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/collablens-match.dir/match_opencv.cpp.i"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rahul/Work/LiveGoggle/Code/popsift/src/application/match_opencv.cpp > CMakeFiles/collablens-match.dir/match_opencv.cpp.i

src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/collablens-match.dir/match_opencv.cpp.s"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rahul/Work/LiveGoggle/Code/popsift/src/application/match_opencv.cpp -o CMakeFiles/collablens-match.dir/match_opencv.cpp.s

src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o: src/application/CMakeFiles/collablens-match.dir/flags.make
src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o: src/application/pgmread.cpp
src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o: src/application/CMakeFiles/collablens-match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rahul/Work/LiveGoggle/Code/popsift/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o -MF CMakeFiles/collablens-match.dir/pgmread.cpp.o.d -o CMakeFiles/collablens-match.dir/pgmread.cpp.o -c /home/rahul/Work/LiveGoggle/Code/popsift/src/application/pgmread.cpp

src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/collablens-match.dir/pgmread.cpp.i"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rahul/Work/LiveGoggle/Code/popsift/src/application/pgmread.cpp > CMakeFiles/collablens-match.dir/pgmread.cpp.i

src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/collablens-match.dir/pgmread.cpp.s"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rahul/Work/LiveGoggle/Code/popsift/src/application/pgmread.cpp -o CMakeFiles/collablens-match.dir/pgmread.cpp.s

# Object files for target collablens-match
collablens__match_OBJECTS = \
"CMakeFiles/collablens-match.dir/match_opencv.cpp.o" \
"CMakeFiles/collablens-match.dir/pgmread.cpp.o"

# External object files for target collablens-match
collablens__match_EXTERNAL_OBJECTS =

Linux-x86_64/collablens-match: src/application/CMakeFiles/collablens-match.dir/match_opencv.cpp.o
Linux-x86_64/collablens-match: src/application/CMakeFiles/collablens-match.dir/pgmread.cpp.o
Linux-x86_64/collablens-match: src/application/CMakeFiles/collablens-match.dir/build.make
Linux-x86_64/collablens-match: Linux-x86_64/libpopsift.so.1.0.0
Linux-x86_64/collablens-match: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Linux-x86_64/collablens-match: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
Linux-x86_64/collablens-match: /usr/lib/x86_64-linux-gnu/libboost_system.so
Linux-x86_64/collablens-match: /usr/local/cuda/lib64/libcudadevrt.a
Linux-x86_64/collablens-match: /usr/lib/x86_64-linux-gnu/libIL.so
Linux-x86_64/collablens-match: /usr/lib/x86_64-linux-gnu/libILU.so
Linux-x86_64/collablens-match: /usr/local/cuda/lib64/libcudart.so
Linux-x86_64/collablens-match: /usr/local/cuda/lib64/libcublas.so
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_gapi.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_stitching.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_alphamat.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_aruco.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_bgsegm.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_bioinspired.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_ccalib.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudabgsegm.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudafeatures2d.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudaobjdetect.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudastereo.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_dnn_superres.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_dpm.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_face.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_freetype.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_fuzzy.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_hdf.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_hfs.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_img_hash.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_intensity_transform.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_line_descriptor.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_mcc.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_quality.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_rapid.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_reg.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_rgbd.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_saliency.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_stereo.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_structured_light.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_superres.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_surface_matching.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_tracking.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_highgui.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_datasets.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_plot.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_text.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_videostab.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_videoio.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudaoptflow.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudalegacy.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudawarping.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_optflow.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_xfeatures2d.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_ml.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_shape.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_ximgproc.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_video.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_dnn.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_xobjdetect.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_imgcodecs.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_objdetect.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_calib3d.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_features2d.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_flann.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_xphoto.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_photo.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudaimgproc.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudafilters.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_imgproc.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudaarithm.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_core.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/lib/libopencv_cudev.so.4.5.2
Linux-x86_64/collablens-match: /usr/local/cuda/lib64/libcudadevrt.a
Linux-x86_64/collablens-match: src/application/CMakeFiles/collablens-match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rahul/Work/LiveGoggle/Code/popsift/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../Linux-x86_64/collablens-match"
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/collablens-match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/application/CMakeFiles/collablens-match.dir/build: Linux-x86_64/collablens-match
.PHONY : src/application/CMakeFiles/collablens-match.dir/build

src/application/CMakeFiles/collablens-match.dir/clean:
	cd /home/rahul/Work/LiveGoggle/Code/popsift/src/application && $(CMAKE_COMMAND) -P CMakeFiles/collablens-match.dir/cmake_clean.cmake
.PHONY : src/application/CMakeFiles/collablens-match.dir/clean

src/application/CMakeFiles/collablens-match.dir/depend:
	cd /home/rahul/Work/LiveGoggle/Code/popsift && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rahul/Work/LiveGoggle/Code/popsift /home/rahul/Work/LiveGoggle/Code/popsift/src/application /home/rahul/Work/LiveGoggle/Code/popsift /home/rahul/Work/LiveGoggle/Code/popsift/src/application /home/rahul/Work/LiveGoggle/Code/popsift/src/application/CMakeFiles/collablens-match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/application/CMakeFiles/collablens-match.dir/depend

