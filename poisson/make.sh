#/bin/bash

make clean
make -s USER_CFLAGS="-DPHG_TO_P4EST -DDim=3" cutfem 
