main=stokes
default: $(main)
$(main): $(main).o 
$(main).o: $(main).c 

clean:
	-/bin/rm -f $(main) *.o 
cleanerr:
	-/bin/rm -f *.err
cleanlog:
	-/bin/rm -f *.log
cleanvtk:
	-/bin/rm -f *.vtk

#include ${PHG_MAKEFILE_INC}
# include /soft/apps/phg/gcc-10.2.0/mvapich2-2.3.5/phg-0.9.6/share/phg/Makefile.inc
# include /opt/phg/phg-0.9.7/share/phg/Makefile.inc
include /opt/phg/0.9.9-20251120/share/phg/Makefile.inc
