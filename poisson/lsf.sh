#!/bin/bash

#BSUB -J CutFEM
#BSUB -n 108
#BSUB -o %J.log
#BSUB -e %J.err
#BSUB -W 6000
#BSUB -q batch
#BSUB -R "span[ptile=18]"

for r0 in 1 2 3 4 5 6 7; do
#  for c2 in 1e-4 1e-3 1e-2 1e-1 1.0 1e+1 1e+2 1e+3 1e+4;do
	mpirun -n 108 ./cutfem -sol_order -100 -refine0 $r0 -dof_type Q1 \
 	-solver mumps -mumps_fillin 500 -solver_cond \
	gp_type 1 -coeff1 1.0 -gamma2 0.0 
#	-coeff2 1.0 +interior_only
#	>> Q2_gp1_$gp.log
#  done
done
