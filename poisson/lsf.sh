#!/bin/bash

#BSUB -J CutFEM
#BSUB -n 288
#BSUB -o %J.log
#BSUB -e %J.err
#BSUB -W 6000
#BSUB -q batch
#BSUB -R "span[ptile=36]"

for r0 in 1 2 3 4 5; do
	mpirun -n 144  ./cutfem -sol_order -100 -refine0 $r0 -dof_type Q4  \
 	-solver mumps -mumps_fillin 600 -solver_cond  \
	-gp_type 1 -coeff1 1.0 -gamma2 0.1 
#	-coeff2 1.0 +interior_only  
done
