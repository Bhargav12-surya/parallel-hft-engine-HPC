# CMake generated Testfile for 
# Source directory: D:/HPC
# Build directory: D:/HPC/build-full
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[native_orderbook_accounting_features]=] "D:/HPC/build-full/hftlab_native_tests.exe")
set_tests_properties([=[native_orderbook_accounting_features]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/HPC/CMakeLists.txt;87;add_test;D:/HPC/CMakeLists.txt;0;")
add_test([=[native_capability_protocol]=] "D:/HPC/build-full/hftlab_runner.exe" "--capabilities")
set_tests_properties([=[native_capability_protocol]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/HPC/CMakeLists.txt;88;add_test;D:/HPC/CMakeLists.txt;0;")
