Configure initial boundary in boundary.cpp
Configure speed function in speed_function.cpp (NOTE: using input files (-icsv, -inpy, -segy) will override the speed function defined in speed_function.cpp)

Set error bound with -e option (default 2e-6).

Commandline examples:

./fmm -ij=100,200 // 100x200 grids, speed function defined in speed_function.cpp

./fmm -ij=13601,2801 -segy=/net/ohm/export/iss/inputs/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy

./fmm -inpy=/net/ohm/export/iss/inputs/elastic-marmousi-model/scaled/p_wave_velocity_x4.npy -h=.3125,.3125 -sf=.001 -rf=1 -t=40

./fmm -algo=fmm -inpy=/net/ohm/export/iss/inputs/elastic-marmousi-model/scaled/p_wave_velocity_x1.npy -h=1.25,1.25 -sf=.001 -dense -strict -rf=1 -t=40

See ./fmm --help for more options.
