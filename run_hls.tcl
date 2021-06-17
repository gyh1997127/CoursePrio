#===================================
# run_hls.tcl for GEMM
#===================================
set SRC_DIR src
# open the HLS project mm.prj
open_project mm.prj -reset
# set the top-level function of the design to be mm
set_top kernel_gemm
# add design files
add_files $SRC_DIR/mm.cpp $SRC_DIR/mm.h
# add the testbench files
add_files -tb $SRC_DIR/mm_test.cpp
# open HLS solution solution1
open_solution "solution1"
# set target FPGA device: Alveo U200 in this example
set_part {xcu200-fsgd2104-2-e}
# target clock period is 5 ns, i.e., 200MHz
create_clock -period 5

# do a c simulation
csim_design
# synthesize the design
csynth_design
# do a co-simulation
#cosim_design
# close project and quit
close_project
# exit Vivado HLS
quit
