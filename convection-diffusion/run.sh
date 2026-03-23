#bash

#!/bin/bash

# refine level
r=5
# so (-1: Trigonometric function, -100: Toroidal flow)
so=-1
# c (1: w(x, y, z) = (1, 0, 0), 2: w(x, y, z) = (-y, x, 0))
c=2

for order in 1;do
for coe in 0.005 0.01 0.1 1.0 100 10000;do 
	mpirun -np 64 ./ConvDiff -refine0 $r -dof_type Q$order -sol_order $so -conv_type $c -coeff1 $coe \
        -solver gmres  -gmres_pc_type solver -gmres_pc_opts "-solver hypre -hypre_solver boomeramg \
       	-solver_maxit 2" -solver_cond -solver_monitor \
	>> cos_cv2_Q$order.log
done
done
