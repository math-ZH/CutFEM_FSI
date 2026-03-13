#/bin/bash

make clean
make USER_CFLAGS="-DPHG_TO_P4EST -DDim=3 -w" ConvDiff
