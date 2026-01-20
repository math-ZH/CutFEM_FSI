/* Parallel Hierarchical Grid -- an adaptive finite element library.
 *
 * Copyright (C) 2005-2010 State Key Laboratory of Scientific and
 * Engineering Computing, Chinese Academy of Sciences. */

/* This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA  02110-1301  USA */

#include <phg.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>	/* isspace */
#include <limits.h>	/* ULONG_MAX */

/*---------------------------------------------------------------------------
 * This program solves the following interface problem using XFEM:
 *	-\div (a \grad u) = f,		in \Omega^i,
 *			u = g_D,	on \partial\Omega_D,
 *		 a\grad u = g_N\cdot n,	on \partial\Omega_N,
 *	plus jump conditions on the interfaces,
 * where \Omega is the computational domain with a finite element mesh,
 * a, f and the solution u are piecewise smooth functions whose smooth
 * regions are nfitted to the mesh, and \Omega^i denotes the union of the
 * smooth regions.
 *
 * For the case of a level-set function, the smooth regions are \Omega^- and
 * \Omega^+ with an interface \Gamma, where:
 * 	\Omega^- := \Omega \cap \{x: L(x)<0\},
 * 	\Omega^+ := \Omega \cap \{x: L(x)>0\},
 * 	\Gamma	 := \Omega \cap \{x: L(x)=0\},
 * 	a := COEFF1 for x\in\Omega^- and COEFF2 for x\in\Omega^+,
 * and L(x) is the level-set function defined by ls_func(). In this case the
 * jump conditions at the interface are given by:
 *		      [u] = j_D,	on \Gamma,
 *	       [a\grad u] = j_N\cdot n,	on \Gamma,
 *
 * For the case of a CAD model (when -CAD_file option is used), the smooth
 * regions consist of the input parts and (optionally) the exterior part.
 *
 * The analytic test solutions are given by func_u().
 *
 * $Id: unfitted.c,v 1.97 2025/09/01 02:24:06 zlb Exp $
 *---------------------------------------------------------------------------*/

/* Note: for cross checking with ipdg.c, the following commands should produce
 * exactly the same results:
 * -------------- For cuboidal meshes
 * % ./ipdg -rel -solver mumps
 * % ./unfitted -xfem_dtol=0 -xfem_etol=0 -xfem_ortho=none -added_order=0 -coeff1=1 -xc=0 -yc=0 -zc=1 -c=3 -ls_order=1 -solver mumps
 * % ./unfitted -coeff1=1 -coeff2=1 -ls_order=1 -added_order=0 -xc=0 -yc=0 -zc=1 -c=0.5 -no_jump -solver mumps
 * -------------- For tetrahedral meshes
 * % ./ipdg -rel -solver mumps -mesh_file=../test/cube.dat -refine0=3
 * % ./unfitted -xfem_dtol=0 -xfem_etol=0 -xfem_ortho=none -added_order=0 -coeff1=1 -xc=0 -yc=0 -zc=1 -c=3 -ls_order=1 -solver mumps -mesh_file=../test/cube.dat -refine0=3
 * % ./unfitted -coeff1=1 -coeff2=1 -ls_order=1 -added_order=0 -xc=0 -yc=0 -zc=1 -c=0.5 -no_jump -solver mumps -mesh_file=../test/cube.dat -refine0=3
 */

/* input MG.h */
#include "inexact_xasp.h"

BOOLEAN use_pc = FALSE;

static BOOLEAN LDG = FALSE;
static int Q_BAS1, Q_BAS2, Q_BAS3;
static MAT *B[Dim];
static VEC *F[Dim];
static FLOAT gammat = 0.0;

/* The coefficients */
static FLOAT coeff[] = {1.0, 2.0};

#if 1
/* Round-robin: 0, 1, 0, 1, 0, 1, ... */
#define Coeff(i)	(coeff[(i) % 2])
#define Func(f)		func_no %= 2; f(x, y, z, res, &func_no);
#else
/* Repeated 0s: 0, 1, 0, 0, 0, 0, ... */
#define Coeff(i)	(coeff[(i) == 1 ? 1 : 0])
#define Func(f)		func_no = 0; f(x, y, z, res, &func_no);
#endif

/* The IP and ghost parameters */
static FLOAT beta = 1.0, gamma0 = 100.0, gamma1 = 0.1, gamma2 = 0.1;
static int gp_type = 1;	/* Ghost penalty type: 0.Face_based, 1.Element_based */

static FLOAT xc = 0.5, yc = 0.5, zc = 0.5, c = 0.3;
static INT ls_order = 2;

/* extra quad. orders for interface elements */
static INT added_order = 7;

/* control parameters */
BOOLEAN no_jump = FALSE;

/* order of analytic solution (>=0 => polynomial), with some special cases:
 * 	-99:	sol_order := DOF_DEFAULT->order
 * 	-100:	Spherical test solution, with:
 *		. u(x,y,z) := \sqrt{(x-xc)^2 + (y-xc)^2 + (z-xc)^2},
 *		. ls_func() is negated (i.e., interior and exterior exchanged),
 *		. the option -interior is forced
 *		. npart is set to 1.
 */
static INT sol_order = 1;

static BOOLEAN dump_solver = FALSE, dump_reorder = TRUE;

/* analytic solution */
/*static*/ int
func_u(FLOAT x, FLOAT y, FLOAT z, FLOAT *res, void *parm)
{
    int func_no = *(int *)parm;

    if (sol_order == -100) {
	assert(func_no == 0);
	x -= xc; y -= yc; z -= zc;
	*res = Sqrt(x * x + y * y + z * z);
	return 0;
    }

    switch (func_no) {
	case 0:
	    if (sol_order < 0)
		*res = Cos(x * y * z);
	    else
		*res =  1.0 + Pow(x, sol_order) -
			2.0 * Pow(y, sol_order) +
			3.0 * Pow(z, sol_order);
	    break;
	case 1:
	    if (sol_order < 0 && !no_jump)
		*res = Sin(x + y + z);
	    else {
		func_no = 0; func_u(x, y, z, res, &func_no);
		if (!no_jump)
		    *res *= 10.0;
	    }
	    break;
	default:
	    Func(func_u)
    }

    return 1;
}

static int
func_grad_u(FLOAT x, FLOAT y, FLOAT z, FLOAT *res, void *parm)
{
    int func_no = *(int *)parm;

    if (sol_order == -100) {
	FLOAT r;
	assert(func_no == 0);
	x -= xc; y -= yc; z -= zc;
	r = Sqrt(x * x + y * y + z * z);
	r = r < FLOAT_EPSILON ? 0.0 : 1.0 / r;
	res[0] = x * r;
	res[1] = y * r;
	res[2] = z * r;
	return 0;
    }

    switch (func_no) {
	case 0:
	    if (sol_order < 0) {
		res[0] = -y * z * Sin(x * y * z);
		res[1] = -x * z * Sin(x * y * z);
		res[2] = -x * y * Sin(x * y * z);
	    }
	    else if (sol_order == 0) {
		res[0] = res[1] = res[2] = 0.;
	    }
	    else {
		res[0] = 1.0 * sol_order * Pow(x, sol_order - 1.0);
		res[1] =-2.0 * sol_order * Pow(y, sol_order - 1.0);
		res[2] = 3.0 * sol_order * Pow(z, sol_order - 1.0);
	    }
	    break;
	case 1:
	    if (sol_order < 0 && !no_jump) {
		res[0] = Cos(x + y + z);
		res[1] = Cos(x + y + z);
		res[2] = Cos(x + y + z);
	    }
	    else {
		func_no = 0; func_grad_u(x, y, z, res, &func_no);
		if (no_jump)
		    return 3;
		res[0] *= 10.0;
		res[1] *= 10.0;
		res[2] *= 10.0;
	    }
	    break;
	default:
	    Func(func_grad_u)
    }

    return 3;
}

/* the jump/boundary functions */
static int
func_gD(FLOAT x, FLOAT y, FLOAT z, FLOAT *res, void *parm)
{
    if (sol_order == -100 && ls_order == 2) {
	FLOAT r2 = (x-xc) * (x-xc) + (y-yc) * (y-yc) + (z-zc) * (z-zc);
	if (r2 < 1.1 * c * c) {
	    assert(r2 < 1.01 * c * c);
	    *res = c;	/* constant */
	    return 0;
	}
    }

    return func_u(x, y, z, res, parm);
}

#define func_gN		func_grad_u

/* right hand side */
static int
func_f(FLOAT x, FLOAT y, FLOAT z, FLOAT *res, void *parm)
{
    int func_no = *(int *)parm;

    if (sol_order == -100) {
	FLOAT r;
	assert(func_no == 0);
	x -= xc; y -= yc; z -= zc;
	r = Sqrt(x * x + y * y + z * z);
	*res = - Coeff(0) * 2.0 / (r < FLOAT_EPSILON ? 1.0 : r);
	return 0;
    }
    switch (func_no) {
	case 0:
	    if (sol_order < 0) {
		*res = +(x * x * y * y + x * x * z * z + y * y * z * z)
						* Cos(x * y * z) * Coeff(0);
	    }
	    else {
		if (sol_order <= 1)
		    *res = -0. * Coeff(0);
		else
		    *res = -sol_order * (sol_order - 1.) * Coeff(0) *
			(1.0 * Pow(x, sol_order - 2.) -
			 2.0 * Pow(y, sol_order - 2.) +
			 3.0 * Pow(z, sol_order - 2.));
	    }
	    break;
	case 1:
	    if (sol_order < 0 && !no_jump) {
		*res = 3. * Sin(x + y + z) * Coeff(1);
	    }
	    else {
		func_no = 0; func_f(x, y, z, res, &func_no);
		*res *= Coeff(1) / Coeff(0);
		if (!no_jump)
		    *res *= 10.0;
	    }
	    break;
	default:
	    Func(func_f)
    }

    return 1;
}

/* level set function */

static void
ls_func(FLOAT x, FLOAT y, FLOAT z, FLOAT *value)
{
    assert(ls_order == 1 || ls_order == 2);
    if (ls_order == 2) {
	/* Sphere */
	x -= xc; y -= yc; z -= zc;
	*value = x * x + y * y + z * z - c * c;
    }
    else {
	/* plane */
	*value = x * xc + y * yc + z * zc - c;
    }
    if (sol_order == -100)
	*value = -*value;
}

static void
ls_grad_func(FLOAT x, FLOAT y, FLOAT z, FLOAT *grad)
{
    assert(ls_order == 1 || ls_order == 2);
    if (ls_order == 2) {
	/* Sphere */
	x -= xc; y -= yc; z -= zc;
	grad[0] = x + x; grad[1] = y + y; grad[2] = z + z;
    }
    else {
	/* plane */
	grad[0] = xc; grad[1] = yc; grad[2] = zc;
    }
    if (sol_order == -100)
	grad[0] = -grad[0], grad[1] = -grad[1], grad[2] = -grad[2];
}

static void
ls_info(GRID *g)
/* prints data for the sphere or the plane to help debugging with ParaView */
{
    assert(ls_order == 1 || ls_order == 2);
    if (ls_order == 1) {
	/* find a parallelogram on the plane enclosing the mesh for ParaView */
#if D
EBUG && defined(PHG_TO_P4EST)
	/* local coordinate system on the plan: (origin, v0, v1) */
	double d, bbox[2][2] = {{0.,0.},{0.,0.}}, origin[Dim], v0[Dim], v1[Dim];
	double nv[] = {xc, yc, zc};
	int i, flag = 0;
	/* normalize nv */
	d = 1.0 / sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
	nv[0] *= d;
	nv[1] *= d;
	nv[2] *= d;
	/* origin := projection of the center */
	origin[0] = (g->bbox[0][0] + g->bbox[1][0]) * 0.5;
	origin[1] = (g->bbox[0][1] + g->bbox[1][1]) * 0.5;
	origin[2] = (g->bbox[0][2] + g->bbox[1][2]) * 0.5;
	/* compute d such that (origin + d*nv).(xc,yc,zc) == c */
	d *= c - origin[0] * xc - origin[1] * yc - origin[2] * zc;
	origin[0] += d * nv[0];
	origin[1] += d * nv[1];
	origin[2] += d * nv[2];
	/* i := the smallest component of (xc,yc,zc) */
	d = Fabs(xc); i = 0;
	if (d > Fabs(yc)) {d = Fabs(yc); i = 1;}
	if (d > Fabs(zc)) {d = Fabs(zc); i = 2;}
	switch (i) {
	    case 0: v0[0] = 0.; v0[1] = nv[2]; v0[2] = -nv[1]; break;
	    case 1: v0[1] = 0.; v0[0] = nv[2]; v0[2] = -nv[0]; break;
	    case 2: v0[2] = 0.; v0[0] = nv[1]; v0[1] = -nv[0]; break;
	}
	d = 1.0 / sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
	v0[0] *= d;
	v0[1] *= d;
	v0[2] *= d;
	/* v1 := nv cross v0 */
	v1[0] = nv[1] * v0[2] - nv[2] * v0[1];
	v1[1] = nv[2] * v0[0] - nv[0] * v0[2];
	v1[2] = nv[0] * v0[1] - nv[1] * v0[0];
	for (i = 0; i < NEdge; i++) {
	    /* find the cut point with the i-th edge */
	    int i0 = GetEdgeVertex(i, 0), i1 = GetEdgeVertex(i, 1);
	    double c0[] = {g->bbox[i0 & 1][0],
			   g->bbox[(i0>>1) & 1][1],
			   g->bbox[(i0>>2) & 1][2]},
		   c1[] = {g->bbox[i1 & 1][0] - c0[0],
			   g->bbox[(i1>>1) & 1][1] - c0[1],
			   g->bbox[(i1>>2) & 1][2] - c0[2]};
	    /* solve c0[].{xc,yc,zc} + t*c1[].{xc,yc,zc} == c for t */
	    double a = c1[0] * xc + c1[1] * yc + c1[2] * zc,
		   b = c - c0[0] * xc - c0[1] * yc - c0[2] * zc;
	    if (a == 0.0)
		continue;
	    if ((d = b / a) < -FLOAT_EPSILON || d > 1 + FLOAT_EPSILON)
		continue;
	    c0[0] += d * c1[0];
	    c0[1] += d * c1[1];
	    c0[2] += d * c1[2];
	    c0[0] -= origin[0];
	    c0[1] -= origin[1];
	    c0[2] -= origin[2];
	    /* (a,b) := coordinates w.r.t. v0 and v1 */
	    a = c0[0] * v0[0] + c0[1] * v0[1] + c0[2] * v0[2];
	    b = c0[0] * v1[0] + c0[1] * v1[1] + c0[2] * v1[2];
	    if (flag == 0) {
		flag = 1;
		bbox[0][0] = bbox[1][0] = a;
		bbox[0][1] = bbox[1][1] = b;
		continue;
	    }
	    if (a < bbox[0][0])
		bbox[0][0] = a;
	    else if (a > bbox[1][0])
		bbox[1][0] = a;
	    if (b < bbox[0][1])
		bbox[0][1] = b;
	    else if (b > bbox[1][1])
		bbox[1][1] = b;
	}
	/* enlarge the bbox by 10% */
	d = (bbox[1][0] - bbox[0][0]) * 0.1;
	bbox[0][0] -= d;
	bbox[1][0] += d;
	d = (bbox[1][1] - bbox[0][1]) * 0.1;
	bbox[0][1] -= d;
	bbox[1][1] += d;
	phgPrintf("The plane:\n    origin: (%g %g %g)\n    "
		  "point 1: (%g %g %g)\n    point 2: (%g %g %g)\n",
			/* origin: (bbox[0][0], bbox[0][1]) */
			origin[0] + bbox[0][0] * v0[0] + bbox[0][1] * v1[0],
			origin[1] + bbox[0][0] * v0[1] + bbox[0][1] * v1[1],
			origin[2] + bbox[0][0] * v0[2] + bbox[0][1] * v1[2],
			/* point1: (bbox[1][0], bbox[0][1]) */
			origin[0] + bbox[1][0] * v0[0] + bbox[0][1] * v1[0],
			origin[1] + bbox[1][0] * v0[1] + bbox[0][1] * v1[1],
			origin[2] + bbox[1][0] * v0[2] + bbox[0][1] * v1[2],
			/* point2: (bbox[0][0], bbox[1][1]) */
			origin[0] + bbox[0][0] * v0[0] + bbox[1][1] * v1[0],
			origin[1] + bbox[0][0] * v0[1] + bbox[1][1] * v1[1],
			origin[2] + bbox[0][0] * v0[2] + bbox[1][1] * v1[2]);
#endif	/* DEBUG && defined(PHG_TO_P4EST) */
    }
    else {
	phgPrintf("Sphere: center=(%g %g %g), radius=%g\n",
				(double)xc, (double)yc, (double)zc, (double)c);
    }
}

#if 0
/* FIXME: linking against the static library libphg.a, or enabling the line
 * below, makes the code noticebly faster (>2x) */
# include "../src/quad-cache.c"
#endif

/* debugging flags */

#if 0
/* debugging a given RHS entry */
#define DFLAG /*(g->nprocs==1 && I==11378) || (g->nprocs==3 && I==3542)*/TRUE
# define DebugRHS if (DFLAG) \
    phgInfo(-1, "%s:%d: E%"dFMT"/f%d/%d<=>E%"dFMT"/f%d/%d, b[%"dFMT"]+=%g\n", \
	    __func__, __LINE__, GlobalElement(g,eno),face,i, \
	    GlobalElement(g,eno1),face1,j, I, val);
#else
# define DebugRHS Unused(eno); Unused(eno1);	/* do nothing */
#endif

#if 0
/* debugging a given Matrix entry */
#define DFLAG (I==496 && J==496)
# define DebugMat if (DFLAG) \
    phgInfo(-1, "%s:%d: E%"dFMT"/f%d/%d<=>E%"dFMT"/f%d/%d, " \
		"A(%"dFMT",%"dFMT")+=%g\n", __func__, __LINE__, \
		GlobalElement(g,eno),face,i, GlobalElement(g,eno1),face1,j, \
		I, J, val);
#else
# define DebugMat Unused(eno); Unused(eno1);	/* do nothing */
#endif

/* function numbers in the QCACHE objects */
static int Q_f, Q_gD, Q_gN;

/* factorial function */
int factorial(unsigned int n) 
{
  if (n == 0 || n == 1) 
    return 1;
  else 
    return n * factorial(n - 1);
}

BOOLEAN active_mesh_bry(int pno, XFEM_INFO *xi, ELEMENT *e, int face)
/* Use for CutFEM: Judge the face of e is belong to boundary of pno-th active mesh */
{
	int i, n = (int)pow(2, Dim - 1);
	FLOAT pt[n][Dim]; /* verteces of face: pt[2^(Dim-1)][Dim] */
	FLOAT value;

	#ifdef PHG_TO_P4EST
	#define c(k,l) e->corners[k][l]
	assert(xi->info[e->index].mark == 0 && Dim == 3);

	/* get verteces of face, and use the level set function to judge */
	for (i = 0; i < n; i++) {
		switch (face) {
	case 0:	case 1:
		pt[i][0] = c(face,0); pt[i][1] = c(i&1, 1); pt[i][2] = c((i >> 1)&1, 2); break;
	case 2:	case 3: 
		pt[i][0] = c(i&1, 0); pt[i][1] = c(face%2, 1); pt[i][2] = c((i >> 1)&1, 2); break;
	case 4:	case 5: 
		pt[i][0] = c(i&1, 0); pt[i][1] = c((i >> 1)&1, 1); pt[i][2] = c(face%4, 2); break;
	default: phgError(1, "%s: invalid face no. (%d)\n", __func__, face);
		}

		// xi->ls->userfunc(pt[i][0], pt[i][1], pt[i][2], value);
		ls_func(pt[i][0], pt[i][1], pt[i][2], &value);
		if (pno == 0 ? value <= 0 : value >= 0)
			return FALSE;
	}

	return TRUE;
  #endif
}

static void
ghost_penalty1(SOLVER *solver, XFEM_INFO *xi, DOF *u_h, 
	ELEMENT *e, int face, ELEMENT *e1, int face1)
/* Implement ghost penalty under specific requrement of faces: face based type */
{
	assert(face >= 0 && face1 >= 0 && e1 != NULL);

	/* if dof_type = DG, don't implement ghost penalty */
	if (u_h->type->fe_space == FE_L2 || gamma2 == 0)
		return;

	GRID *g = xi->g;
  int n, n1, p;   /* p = polynomial order */
  INT I, J;
  int i, j, k;
  FLOAT val, a, nv[Dim], *rule;
	int eno = e->index, eno1 = e1->index;
	FLOAT h, G2; /* G2 = coeff*gamma2*h^(2k-1)/[(k-1)!]*/
	QCACHE *qc;

	/* switch (e,face) and (e1,face1) such that interface cuts element e */
	if (xi->info[eno].mark != 0) {
		ELEMENT *tmp = e; e = e1; e1 = tmp;
		i = face; face = face1; face1 = i;
		j = eno; eno = eno1; eno1 = j;
		if (xi->info[eno].mark != 0)
			return;
	}

	qc = phgQCNew(QD_DEFAULT, u_h);
	n = n1 = DofNBas(u_h, e);
  p = DofTypeOrder(u_h, e);
  h = phgGeomGetFaceDiameter(g, e, face);

#if Dim == 3
    rule = phgQuadGetRule2D(g, e, face, 2 * p);
#else
    rule = phgQuadGetRule1D(g, e, face, 2 * p);
#endif
	phgQCSetRule(qc, rule, -1.);
  phgGeomGetFaceOutNormal(g, e, face, nv);
  phgQCSetConstantNormal(qc, nv);

#define Sel(i, o, o1)	(i < n ? o : o1)
#define Ele(i)	Sel(i, e, e1)
#define Fac(i)	Sel(i, face, face1)
#define Bas(i)	Sel(i, i, i - n)
#define Eid(i)	Ele(i)->index
#define Quad(fid1, proj1, i1, fid2, proj2, i2) \
	    phgQCIntegrateFace( \
				qc, Eid(i1), Fac(i1), fid1, proj1, Bas(i1), \
				qc, Eid(i2), Fac(i2), fid2, proj2, Bas(i2))

	/* loop on {basis funcs in e} \cup {basis funcs in e1} */
  for (i = 0; i < n + n1; i++) {
	/* loop on {basis funcs in e} \cup {basis funcs in e1} */
		for (j = 0; j < n + n1; j++) {

	  /* skip jumps for interior face and continuous element */
	  	if (u_h->type->fe_space == FE_H1
#ifdef PHG_TO_P4EST
		/* e, e1 are equal size (unless dependent DOFs are removed) */
		&& e->generation == e1->generation
#endif	/* defined(PHG_TO_P4EST) */
			)	{
			/* -----------------------------------------------------------------------
			 * Face ghost penalty: interior face and face covered by interface
			 * ----------------------------------------------------------------------*/
				val = 0.0;
				/* sum_{k=1}^p \int G2 [\grad^k u][\grad^k v] */
				for (k = 1; k <= p; k++) {
					if (xi->nd == 1)
						G2 = coeff[0] * coeff[1] * (1.0 / ( coeff[0] + coeff[1])) * gamma2
												* (FLOAT)pow(h, 2*k-1) / factorial(k-1);
					if (xi->nd == 0)
						G2 = 0.5 * coeff[0] * gamma2 * (FLOAT)pow(h, 2*k-1) / factorial(k-1);
					a = Quad(Q_GRAD,  PROJ_DOT, j, Q_GRAD,  PROJ_DOT, i);
					if ((i < n) != (j < n)) 
							a = -a;
					val += G2 * a;
				}
				/* If face is not boundary of pno-th active mesh,
					 then implement ghost penalty of space V_{h,pno} */
				for (int pno = 0; pno < xi->nd; pno++) {
					if (!active_mesh_bry(pno, xi, e, face)) {
						I = phgXFEMSolverMapE2G(xi, solver, 0, pno, Ele(i), Bas(i));
						J = phgXFEMSolverMapE2G(xi, solver, 0, pno, Ele(j), Bas(j));
						phgSolverAddGlobalMatrixEntry(solver, I, J, val);
					}
				}
			}
		}
}
#undef Sel
#undef Ele
#undef Fac
#undef Bas
#undef Eid
#undef Quad
}

static void
ghost_penalty2(SOLVER *solver, XFEM_INFO *xi, DOF *u_h, 
	ELEMENT *e, int face, ELEMENT *e1, int face1)
/* Implement ghost penalty under specific requrement of faces: element based type */
{		
	assert(face >= 0 && face1 >= 0 && e1 != NULL);

	/* if dof_type = DG, don't implement ghost penalty */
	if (u_h->type->fe_space == FE_L2 || gamma2 == 0) {
		phgPrintf("Dof_type == DG, don't implement ghost penalty!\n");
		return;
	}

	GRID *g = xi->g;
  int n, n1, p;   /* p = polynomial order */
  INT I, J;
  int i, j, k;
  FLOAT val, nv[Dim], *rule;
	int eno = e->index, eno1 = e1->index;
	FLOAT h, G2; /* G2 = coeff*gamma2*h^(2k-1)/[(k-1)!]*/
	QCACHE *qc;

	/* switch (e,face) and (e1,face1) such that interface cuts element e */
	if (xi->info[eno].mark != 0) {
		ELEMENT *tmp = e; e = e1; e1 = tmp;
		i = face; face = face1; face1 = i;
		j = eno; eno = eno1; eno1 = j;
		if (xi->info[eno].mark != 0) {
			phgPrintf("Non-cut patch, don't implement ghost penalty!\n");
			return;
		}
	}

	phgPrintf("\nGhost_penalty on (elem_%d,face_%d), (elem_%d,face1_%d)\n", eno, face, eno1, face1);

	qc = phgQCNew(QD_DEFAULT, u_h);
	n = n1 = DofNBas(u_h, e);
  p = DofTypeOrder(u_h, e);
  h = phgGeomGetFaceDiameter(g, e, face);
	if (xi->nd == 1)
		G2 = coeff[0] * coeff[1] * (1.0 / (coeff[0] + coeff[1])) * gamma2 / (h * h);
	else if (xi->nd == 0)
		G2 = 0.5 * coeff[0] * gamma2 / (h * h);

/* Now interface cuts element e, and patch(e,e1) */
ELEMENT *e_list[2]={e, e1};
/* If face is not boundary of pno-th active mesh,
 * then implement ghost penalty of space V_{h,pno} */
for (int pno = 0; pno < xi->nd; pno++) {
	if (!active_mesh_bry(pno, xi, e, face)) {
		phgPrintf("(e_%d, face_%d) is not bdry of %d-th active mesh\n", eno, face, pno);

		for (int k = 0; k < 2; k++) {
#if Dim == 3
	rule = phgQuadGetRule3D(g, e_list[k], 2 * DofTypeOrder(u_h, e_list[k]) /*- 2*/);
#else 
	rule = phgQuadGetRule2D(g, e_list[k], 2 * DofTypeOrder(u_h, e_list[k]) /*- 2*/);
#endif
	phgQCSetRule(qc, rule, -1.);

#define Sel(i, o, o1)	(i < n ? o : o1)
#define Ele(i)	Sel(i, e, e1)
#define Fac(i)	Sel(i, face, face1)
#define Bas(i)	Sel(i, i, i - n)
#define Eid(i)	Ele(i)->index

	/* loop on {basis funcs in e} \cup {basis funcs in e1} */
	for (i = 0; i < n + n1; i++) {
		/* loop on {basis funcs in e} \cup {basis funcs in e1} */
		for (j = 0; j < n + n1; j++) {
			val = 0.0;
			/* \int_{e or e1} G2 [u^e][v^e] */
			val = phgQCIntegrate(qc, Eid(j), Q_BAS, j, qc, Eid(i), Q_BAS, i);
			if ((i < n) != (j < n))
				val = -val;
			// debug
			if (k == 0) {
				for (int m = 0; m < sizeof(rule) / sizeof(rule[0]); m++) {
					phgPrintf("%f ", (double)rule[m]);
				}
				// phgPrintf("Rule on elem_%d, bas on (elem_%d, elem_%d), (bas_%d, bas_%d) = %e\n",
				// 		e_list[0]->index, Eid(i), Eid(j), i, j, val);
				phgPrintf("\n");
			}
			val += G2 * val;
					I = phgXFEMSolverMapE2G(xi, solver, 0, pno, Ele(i), Bas(i));
					J = phgXFEMSolverMapE2G(xi, solver, 0, pno, Ele(j), Bas(j));
					phgSolverAddGlobalMatrixEntry(solver, I, J, val);
				}
			}
		}
	}
}
#undef Sel
#undef Ele
#undef Fac
#undef Bas
#undef Eid
#undef Quad
}

static void
do_face(SOLVER *solver, XFEM_INFO *xi, QCACHE *qc[],
	ELEMENT *e, int face, int pno, ELEMENT *e1, int face1, int pno1,
	/* for debugging */int line)
/* This function computes integrations on a surface area, which can be:
 *
 *   1) An element face cut by a part, in this case:
 * 	"e" != "e1" && "pno" == "pno1", and "e1" == NULL <=> boundary face.
 *
 *   2) A surface cut by an element face, in this case:
 *   	"e" != "e1" && "e1" != NULL.
 *
 *   3) A surface cut by an element, in this case:
 *	"e" == "e1" && "face" == "face1" == -1. 
 *
 * The normal vectors in the quadrature rule attached to "qc[]" must point
 * from "pno" to "pno1".
 */
{
    GRID *g = xi->g;
    DOF *u;
    int n, n1, p;   /* p = polynomial order */
    INT I, J, eno = e->index, eno1 = e1 == NULL ? -1 : e1->index;
    int i, j;
    FLOAT val;
    FLOAT G0, G1, h, a, b; /* G0=coeff*gamma0*p^2/h, G1=coeff*gamma1*h/p^2 */
    FLOAT Gt;
    BTYPE bdry_flag;

    if (pno == xi->nd) { /* pno is the exterior part */
	/* switch (e,face,pno) and (e1,face1,pno1) such that pno < xi->npart */
	FLOAT *rule, *nv;
	ELEMENT *tmp = e; e = e1; e1 = tmp;
	i = face; face = face1; face1 = i;
	i = pno; pno = pno1; pno1 = i;
	if (pno >= xi->nd)
	    return;		/* both parts are outside */
	/* reverse the normal direction of the rule.
	 * Note: no need to preserve the rule since it's used only once. */
	rule = (FLOAT *)phgQCGetRule(qc[pno]);
	n = phgQIRuleInfo(rule, NULL, NULL, (const FLOAT **)&nv);
	for (i = 0; i < n; i++, nv += Dim)
	    for (j = 0; j < Dim; j++)
		nv[j] = -nv[j];
    }

    if (phgQCGetNP(qc[pno]) == 0)
	return;

    u = qc[pno]->fe;
    n = phgQCGetNBas(qc[pno], e->index);
    p = phgQCGetOrder(qc[pno], e->index);

    h = e == e1 ? phgGeomGetDiameter(g, e) :
		  phgGeomGetFaceDiameter(g, e, face);

    if (e1 == NULL || pno1 >= xi->nd) {
	/* boundary face */
	if (e1 == NULL)
	    bdry_flag = GetFaceBTYPE(g, e, face);
	else
	    bdry_flag = UNDEFINED; /* TODO: BC type on immersed boundary */
	if (bdry_flag != NEUMANN)
	    bdry_flag = DIRICHLET;
	n1 = 0;
	
	G0 = Coeff(pno) * gamma0 * p * p / h;
	G1 = Coeff(pno) * gamma1 * h / (p * (FLOAT)p);
	Gt = Coeff(pno) * gammat * h / (p * (FLOAT)p);
	/* RHS */
	for (i = 0; i < n; i++) {
	    I = phgXFEMSolverMapE2G(xi, solver, 0, pno, e, i);

	    if (bdry_flag == DIRICHLET) {	/* Dirichlet boundary */
		/* -\beta\int_\Gamma_D g_D (A\grad v).n */
		a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gD,   PROJ_NONE, 0,
				qc[pno], e->index, face, Q_GRAD, PROJ_DOT,  i)
				* Coeff(pno);
		val = -beta * a;

		/* G0 \int_\Gamma_D g_D v */
		a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
				qc[pno], e->index, face, Q_BAS, PROJ_NONE, i);
		val += G0 * a;
		/* Add tangential penalty term on curved boundary/interface */
		if (Gt != 0. /*&& e1 != NULL*/) {
		    /* Gt \int_\Gamma_D gNxn, (\grad v)xn  */
		    a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gN, PROJ_CROSS, 0,
				qc[pno], e->index, face, Q_GRAD, PROJ_CROSS, i);
		    val += Gt * a;
		}

		if (LDG) {
		    /* lifting for Dirichlet boundary */
		    /* \int_\Gamma_D [v] {u e_p \cdot n} */
		    a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
				qc[pno], e->index, face, Q_BAS1, PROJ_DOT, i);
		    phgVecAddGlobalEntry(F[0], 0, I, a);

		    a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
				qc[pno], e->index, face, Q_BAS2, PROJ_DOT, i);
		    phgVecAddGlobalEntry(F[1], 0, I, a);

		    a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
				qc[pno], e->index, face, Q_BAS3, PROJ_DOT, i);
		    phgVecAddGlobalEntry(F[2], 0, I, a);
		}
	    }
	    else {				/* Neumann boundary */
		/* \int_\Gamma_N A g_N v */
		val = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gN,  PROJ_DOT,  0,
				qc[pno], e->index, face, Q_BAS, PROJ_NONE, i)
				* Coeff(pno);
		/* \int_\Gamma_N G1 A g_N (A\grad v).n */
		a = phgQCIntegrateFace(
				qc[pno], e->index, face, Q_gN,   PROJ_DOT, 0,
				qc[pno], e->index, face, Q_GRAD, PROJ_DOT, i)
				* Coeff(pno) * Coeff(pno);
		val += G1 * a;
	    }

	    phgSolverAddGlobalRHSEntry(solver, I, val);
	    DebugRHS
	}
    }
    else {
	assert(phgQCGetRule(qc[pno]) == phgQCGetRule(qc[pno1]));
	bdry_flag = INTERIOR;
	n1 = phgQCGetNBas(qc[pno1], e1->index);
	i = phgQCGetOrder(qc[pno1], e1->index);
	if (p < i)
	    p = i;
	G0 = 0.5 * (Coeff(pno) + Coeff(pno1)) * gamma0 * p * (FLOAT)p / h;
	G1 = 0.5 * (Coeff(pno) + Coeff(pno1)) * gamma1 * h / (p * (FLOAT)p);
	Gt = 0.5 * (Coeff(pno) + Coeff(pno1)) * gammat * h / (p * (FLOAT)p);
    }

#define Dir(i) 0.5 /* Mutiple choice: 0 or 1 for sparsity,
		     * harmonic mean for large coeff(pno)/coeff(pno1) ratio */
#define Sel(i, o, o1)	(i < n ? o : o1)
#define Dof(i)	Sel(i, pno, pno1)
#define Qc(i)	qc[Dof(i)]
#define Coe(i)	Coeff(Dof(i))
#define Ele(i)	Sel(i, e, e1)
#define Fac(i)	Sel(i, face, face1)
#define Bas(i)	Sel(i, i, i - n)
#define Eid(i)	Ele(i)->index
#define Quad(fid1, proj1, i1, fid2, proj2, i2) \
	    phgQCIntegrateFace( \
		Qc(i1), Eid(i1), Fac(i1), fid1, proj1, Bas(i1), \
		Qc(i2), Eid(i2), Fac(i2), fid2, proj2, Bas(i2))

    /* loop on {basis funcs in e} \cup {basis funcs in e1} */
    for (i = 0; i < n + n1; i++) {
			I = phgXFEMSolverMapE2G(xi, solver, 0, Dof(i), Ele(i), Bas(i));
	/* loop on {basis funcs in e} \cup {basis funcs in e1} */
	for (j = 0; j < n + n1; j++) {
	  	J = phgXFEMSolverMapE2G(xi, solver, 0, Dof(j), Ele(j), Bas(j));

	    /* skip jumps for interior face and continuous element */
	    if (u->type->fe_space == FE_H1 && e != e1 && e1 != NULL
#ifdef PHG_TO_P4EST
		/* e, e1 are equal size (unless dependent DOFs are removed) */
		&& e->generation == e1->generation
		/* e and e1 are not merged */
		&& GetMacro(xi, e->index, pno) < 0
		&& GetMacro(xi, e1->index, pno1) < 0
#endif	/* defined(PHG_TO_P4EST) */
			/* the same sign in e and e1 */
		&& pno == pno1)
				continue;

	    /*-----------------------------------------------------------------
	     * Note if the normal vector is reversed, the sign of [.] changes,
	     * while the sign of {.} is unaffected.
	     *----------------------------------------------------------------*/
		
	    val = 0.0;

	    if (bdry_flag != NEUMANN) {
		/* -\int {A\grad u}.n [v] (i<=>v, j<=>u, n<=>e) */
		a = Quad(Q_GRAD,  PROJ_DOT, j, Q_BAS, PROJ_NONE, i) * Coe(j);
		if (bdry_flag == INTERIOR)
		    a *= (i < n ? 0.5 : -0.5);
		val = -a;

		/* -\int \beta [u]{A\grad v}.n (note: i<=>v, j<=>u, n<=>e) */
		a = Quad(Q_BAS, PROJ_NONE, j, Q_GRAD,  PROJ_DOT, i) * Coe(i);
		if (bdry_flag == INTERIOR)
		    a *= (j < n ? 0.5 : -0.5);
		val += -beta * a;

		/* \int G0 [u][v] (i<=>v, j<=>u, n<=>e) */
		a = Quad(Q_BAS, PROJ_NONE, j, Q_BAS, PROJ_NONE, i);
		if (bdry_flag == INTERIOR && (i < n) != (j < n))
		    a = -a;
		val += G0 * a;
		if (Gt != 0. && (pno != pno1 || bdry_flag == DIRICHLET)) {
		    /* \int Gt \int_\Gamma_D [gNxn], [(\grad v)xn],
		     * (i<=>v, j<=>u, n<=>e) */
		    a = Quad(Q_GRAD, PROJ_CROSS, j, Q_GRAD, PROJ_CROSS, i);
		    if (bdry_flag == INTERIOR && (i < n) != (j < n))
			a = -a;
		    val += Gt * a;
		}
	    }

	    if (bdry_flag != DIRICHLET) {
		/* \int G1 [A\grad u].n [A\grad v].n */
		a = Quad(Q_GRAD,  PROJ_DOT, j, Q_GRAD,  PROJ_DOT, i)
			* Coe(j) * Coe(i);
		if (bdry_flag == INTERIOR && (i < n) != (j < n))
		    a = -a;
		val += G1 * a;
	    }

	    phgSolverAddGlobalMatrixEntry(solver, I, J, val);
	    DebugMat

	    if (LDG) {
		/* B[0]: \int [\base u] {\bas v e_0}.n */
		a = Quad(Q_BAS, PROJ_NONE, j, Q_BAS1, PROJ_DOT, i);
		if (bdry_flag == DIRICHLET)
		    val = a;
		else if (bdry_flag == INTERIOR)
		    val = a * (j < n ? Dir(i) : -Dir(i));
		else
		    val = 0.;
		phgMatAddGlobalEntry(B[0], I, J, val);
		
		/* B[1]: \int [\base u] {\bas v e_1}.n */
		a = Quad(Q_BAS, PROJ_NONE, j, Q_BAS2, PROJ_DOT, i);
		if (bdry_flag == DIRICHLET)
		    val = a;
		else if (bdry_flag == INTERIOR)
		    val = a * (j < n ? Dir(i) : -Dir(i));
		else
		    val = 0.;
		phgMatAddGlobalEntry(B[1], I, J, val);
		
		/* B[2]: \int [\base u] {\bas v e_2}.n */
		a = Quad(Q_BAS, PROJ_NONE, j, Q_BAS3, PROJ_DOT, i);
		if (bdry_flag == DIRICHLET)
		    val = a;
		else if (bdry_flag == INTERIOR)
		    val = a * (j < n ? Dir(i) : -Dir(i));
		else
		    val = 0.;
		phgMatAddGlobalEntry(B[2], I, J, val);
	    }
	}

	if (e1 == NULL || pno1 >= xi->nd || pno == pno1)
	    continue;		/* non interface face */

	/* The face is part of an interface, apply jump conditions */
#if 0
	/* only true when npart == 1 */
	assert(xi->info[e->index].mark * xi->info[e1->index].mark <= 0);
#endif

	/* \int [A*jN.n] {v} */
	a = phgQCIntegrateFace(qc[pno],  e->index,  face,  Q_gN,  PROJ_DOT,  0,
			       Qc(i), Eid(i), Fac(i), Q_BAS, PROJ_NONE, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gN,  PROJ_DOT,  0,
			       Qc(i), Eid(i), Fac(i), Q_BAS, PROJ_NONE, Bas(i));
	a = Coeff(pno) * a - Coeff(pno1) * b;
	val = a * 0.5;

	/* G1 \int [A*jN.n] [(A\grad v).n] */
	a = phgQCIntegrateFace(qc[pno],  e->index,  face,  Q_gN,   PROJ_DOT, 0,
			       Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_DOT, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gN,   PROJ_DOT, 0,
			       Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_DOT, Bas(i));
	a = Coeff(pno) * a - Coeff(pno1) * b;
	val += Coe(i) * G1 * (i < n ? a : -a);

	/* -beta \int [jD] {(A\grad v).n} (func[1] := gD) */
	a = phgQCIntegrateFace(qc[pno],  e->index,  face,  Q_gD,   PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_DOT, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gD,   PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_DOT, Bas(i));
	a -= b;
	val += -beta * 0.5 * Coe(i) * a;

	/* G0 \int [jD] [v] */
	a = phgQCIntegrateFace(qc[pno],  e->index,  face,  Q_gD,  PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS, PROJ_NONE, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gD,  PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS, PROJ_NONE, Bas(i));
	a -= b;
	val += G0 * (i < n ? a : -a);

	if (Gt != 0.) {
	    /* Gt \int [jNxn], [(\grad v)xn] */
	    a = phgQCIntegrateFace(qc[pno], e->index, face, Q_gN, PROJ_CROSS, 0,
			Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_CROSS, Bas(i));
	    b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gN, PROJ_CROSS,
			0, Qc(i), Eid(i), Fac(i), Q_GRAD, PROJ_CROSS, Bas(i));
	    a = (a - b);
	    val += Gt * (i < n ? a : -a);
	}

	phgSolverAddGlobalRHSEntry(solver, I, val);
	DebugRHS

	if (!LDG)
	    continue;

	/* lifting for (Dirichlet) jump data on curved interface */
	/* \int j_D {u e_p \cdot n} */
	a = phgQCIntegrateFace(qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS1, PROJ_DOT, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS1, PROJ_DOT, Bas(i));
	a -= b;
	val = a * Dir(i);
	phgVecAddGlobalEntry(F[0], 0, I, val);

	a = phgQCIntegrateFace(qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS2, PROJ_DOT, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS2, PROJ_DOT, Bas(i));
	a -= b;
	val = a * Dir(i);
	phgVecAddGlobalEntry(F[1], 0, I, val);

	a = phgQCIntegrateFace(qc[pno], e->index, face, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS3, PROJ_DOT, Bas(i));
	b = phgQCIntegrateFace(qc[pno1], e1->index, face1, Q_gD, PROJ_NONE, 0,
			       Qc(i), Eid(i), Fac(i), Q_BAS3, PROJ_DOT, Bas(i));
	a -= b;
	val = a * Dir(i);
	phgVecAddGlobalEntry(F[2], 0, I, val);
    }
#undef Sel
#undef Qc
#undef Ele
#undef Fac
#undef Bas
#undef Dof
#undef Coe
#undef Quad
#undef Dir
}

static void
process_mass_matrix(MAT *A)
/* process empty row */
{
    INT i;
    MAT_ROW *row;

    phgMatAssemble(A);
    phgMatDisassemble(A);
    assert(A->type == PHG_UNPACKED);
    for (i = 0; i < A->rmap->nlocal; i++) {
	row = A->rows + i;
	if (row->ncols == 0) {
	    /* empty row */
	    row->ncols = row->alloc = 1;
	    phgFree(row->cols);
	    row->cols = phgAlloc(sizeof(*row->cols));
	    row->cols[0] = i;
	    phgFree(row->data);
	    row->data = phgAlloc(sizeof(*row->data));
	    row->data[0] = 1.0;
	}
    }
    phgMatAssemble(A);
}

static void
build_linear_system(XFEM_INFO *xi, SOLVER *solver, DOF *u_h[], DOF *f_h[])
{
    GRID *g = u_h[0]->g;
    QCACHE **qc;
    ELEMENT *e;
    int k;
    /* for local DG */
    FLOAT e1_data[]={1.,0.,0.}, e2_data[]={0.,1.,0.}, e3_data[]={0.,0.,1.};
    MAT *M, *C, *tmp; /* mass matrix M, coefficient matrix C, */

    qc = phgXFEMQCNew(xi, u_h);
    for (k = 0; k < xi->nd; k++) {	
	/* Note: same values for Q_{f,gD,gN} should be got with different k */
	Q_f = phgQCAddFEFunction(qc[k], f_h[k]);
	Q_gD = phgQCAddXYZFunctionP(qc[k], func_gD, 1, &k, sizeof(k));
	Q_gN = phgQCAddXYZFunctionP(qc[k], func_gN, 3, &k, sizeof(k));
	if (!LDG)
	    continue;
	Q_BAS1 = phgQCAddConstantCoefficient(qc[k], e1_data, Dim, Q_BAS);
	Q_BAS2 = phgQCAddConstantCoefficient(qc[k], e2_data, Dim, Q_BAS);
	Q_BAS3 = phgQCAddConstantCoefficient(qc[k], e3_data, Dim, Q_BAS);
    }

    if (LDG) {
	M = phgMapCreateMat(solver->mat->rmap, solver->mat->cmap);
	M->handle_bdry_eqns = FALSE;
	C = phgMapCreateMat(solver->mat->rmap, solver->mat->cmap);
	C->handle_bdry_eqns = FALSE;
	for (k = 0; k < Dim; k++) {
	    B[k] = phgMapCreateMat(solver->mat->rmap, solver->mat->cmap);
	    B[k]->handle_bdry_eqns = FALSE;
	    F[k] = phgMapCreateVec(solver->mat->cmap, 1);
	    phgVecDisassemble(F[k]);
	}
	ForAllElements(g, e) {
	    int i, N;
	    INT I;
	    FLOAT val;
	    for (k = 0; k < xi->nd; k++) {
		N = phgQCGetNBas(qc[k], e->index);
		for (i = 0; i < N; i++) {
		    I = phgXFEMSolverMapE2G(xi, solver, 0, k, e, i);
		    val = Coeff(k);
		    phgMatAddGlobalEntry(C, I, I, val);
		}
	    }
	}
    }

#if USE_OMP
# pragma omp parallel for private(e)
#endif	/* USE_OMP */
    ForAllElementsBegin(g, e) {
	RULE_LIST *rl;
	INT N, I, J, eno = e->index, eno1 = -1;
	int i, j, face = -1, face1 = -1, nr;
	FLOAT val;

	/*==================== element ====================*/
	nr = phgXFEMGetRules(xi, eno, -1, RL_ALL, &rl);
	for (k = 0; k < nr; k++) {
	    if (rl[k].iflag) {	/* interface rule */
		if (rl[k].pno1 == rl[k].pno)
		    continue;	/* false/dummy interface (can apply inner BC) */
		if (rl[k].pno < xi->nd)
		    phgQCSetRule(qc[rl[k].pno], rl[k].rule, -1.);
		if (rl[k].pno1 < xi->nd)
		    phgQCSetRule(qc[rl[k].pno1], rl[k].rule, -1.);
		do_face(solver, xi, qc, e, -1, rl[k].pno,
					e, -1, rl[k].pno1, __LINE__);
		continue;
	    }
	    /* volume rule */
	    assert(rl[k].pno < xi->nd);
	    phgQCSetRule(qc[rl[k].pno], rl[k].rule, -1.);
	    N = phgQCGetNBas(qc[rl[k].pno], e->index);
	    for (i = 0; i < N; i++) {
		I = phgXFEMSolverMapE2G(xi, solver, 0, rl[k].pno, e, i);
		for (j = 0; j <= i; j++) {
		    /* \int_T A\grad u . \grad v */
		    J = phgXFEMSolverMapE2G(xi, solver, 0, rl[k].pno, e, j);
		    val = phgQCIntegrate(qc[rl[k].pno], eno, Q_GRAD, j,
					 qc[rl[k].pno], eno, Q_GRAD, i)
			  * Coeff(rl[k].pno);
		    phgSolverAddGlobalMatrixEntry(solver, I, J, val); 
	    	    DebugMat
		    if (i != j) {
			phgSolverAddGlobalMatrixEntry(solver, J, I, val); 
	    		DebugMat
		    }
		    if (!LDG)
			continue;
		    /* \int_T \bas u . \bas v */
		    val = phgQCIntegrate(qc[rl[k].pno], eno, Q_BAS, j,
						 qc[rl[k].pno], eno, Q_BAS, i);
		    phgMatAddGlobalEntry(M, I, J, val);
		    DebugMat
		    if (i != j)
			phgMatAddGlobalEntry(M, J, I, val);
		}

		/* \int_T f1 v */
		val = phgQCIntegrate(qc[rl[k].pno], eno, Q_BAS, i,
				     qc[rl[k].pno], eno, Q_f, 0);
		phgSolverAddGlobalRHSEntry(solver, I, val);
		DebugRHS
	    }
	}
	phgQIRuleListFree(nr, &rl);

	/* the faces of the element */
	for (face = 0; face < NFace; face++) {
	    ELEMENT *e1 = phgGetNeighbour(g, e, face);
	    face1 = -1;
	    eno1 = -1;
	    if (e1 != NULL) {
		/* a face is only processed by the smaller of the two
		 * neighbouring elements, and by the one with smaller
		 * global index if the two elements are of the same size,
		 * to avoid double counting and redundant computation */
		if (e->generation < e1->generation)
		    continue;
		if (e->generation == e1->generation &&
		    GlobalElement(g, eno) > GlobalElement(g, e1->index))
		    continue; /* process each interior face just once */
		face1 = phgOppositeFace(g, e, face, e1);
		eno1 = e1->index;
		/* face is not on boundary, then probablly implement ghost penalty */
		if (gamma2 != 0)
			if (gp_type)	/* Element_based type */
				ghost_penalty2(solver, xi, u_h[0], e, face, e1, face1);
			else /* Face_based type */
				ghost_penalty1(solver, xi, u_h[0], e, face, e1, face1);
	    }
	    nr = phgXFEMGetRulesFace(xi, eno, face, -1, &rl);
	    /*assert(nr > 0 || xi->nd <= xi->npart || xi->g_tet != NULL);*/
	    for (k = 0; k < nr; k++) {
		if (rl[k].pno < xi->nd)
		    phgQCSetRule(qc[rl[k].pno], rl[k].rule, -1.);
		if (rl[k].iflag) {
		    /* interface part */
#if 0
		    if (e1 == NULL)
			phgError(1, "%s:%d: forbidden case: surface overlaps "
				"with boundary.\n", __FILE__, __LINE__);
#endif
		    if (rl[k].pno1 < xi->nd)
			phgQCSetRule(qc[rl[k].pno1], rl[k].rule, -1.);
		}
		do_face(solver, xi, qc, e,  face,  rl[k].pno,
					e1, face1, rl[k].pno1, __LINE__);
	    }
	    phgQIRuleListFree(nr, &rl);
	}
    } ForAllElementsEnd

#ifdef PHG_TO_P4EST
    /* count interface DOF */
    MAP *map = solver->rhs->map;
    INT ndof = map->P == NULL ? map->nlocal : map->P->cmap->nlocal;
    char *mark = phgCalloc(ndof, sizeof(*mark));
    INT n0 = solver->rhs->map->partition[g->rank];
    ForAllElements(g, e) {
	for (int k = 0; k < xi->nd; k++) {
	    int N;
	    if ((xi->info[e->index].mark != 0 ||
		 xi->info[e->index].data[k].mark != 0) &&
		GetMacro(xi, e->index, k) < 0)
		continue;   /* non-interface element */
	    N = phgQCGetNBas(qc[k], e->index);
	    for (int i = 0; i < N; i++) {
		INT I = phgXFEMSolverMapE2G(xi, solver, 0, k, e, i) - n0;
		if (I < 0 || I >= solver->rhs->map->nlocal)
		    continue;
		if (map->P_x2y != NULL && (I = map->P_x2y[I]) < 0)
		    continue;
		mark[I] = 1;
	    }
	}
    }
    n0 = 0;
    for (INT I = 0; I < ndof; I++)
	n0 += mark[I];
    phgFree(mark);
    /* count # elements by discretization order */
    int order_max = phgXFEMDofOrder(xi, u_h[0], NULL, 0);
    if (order_max < u_h[0]->type->order)
	order_max = u_h[0]->type->order;
    INT cnts[2 * (order_max + 1)], *cnts0 = cnts + order_max + 1;
    memset(cnts, 0, sizeof(cnts));
    if (xi->g_mac != NULL) {
	ForAllElements(xi->g_mac, e) {
	    ELEMENT *e0 = g->elems[xi->mlist[e->index][0]];
	    int pno;
	    for (pno = 0; pno < xi->nd; pno++)
		if (GetMacro(xi, e0->index, pno) == e->index)
		    break;
	    assert(pno < xi->nd);
	    cnts[phgXFEMDofOrder(xi, u_h[pno], e0, pno)]++;
	    cnts[0]++;	    /* total count */
	}
    }
    ForAllElements(g, e)
	for (int pno = 0; pno < xi->nd; pno++)
	    if (GetMacro(xi, e->index, pno) < 0 &&
		!phgXFEMOffside(xi, e->index, pno)) {
		cnts0[phgXFEMDofType_(xi, u_h[0], e, -1, FALSE, NULL)->order]++;
		cnts0[0]++; /* total count */
	}
# if USE_MPI
    if (g->nprocs > 1) {
	INT c[2 * (order_max + 1)];
	cnts[0] = n0;
	MPI_Reduce(cnts, c, 2 * (order_max + 1), PHG_MPI_INT, MPI_SUM,
		   0, g->comm);
	memcpy(cnts, c, sizeof(c));
	n0 = cnts[0];
    }
# endif /* USE_MPI */
    phgPrintf("  # cut elements by FE order:");
    for (int k = 1; k <= order_max; k++)
	if (cnts[k] != 0)
	    phgPrintf(" %d:%"dFMT, k, cnts[k]);
    phgPrintf("%s\n", cnts[0] ? "" : " none");
    phgPrintf("  # non-cut elements by FE order:");
    for (int k = 1; k <= order_max; k++)
	if (cnts0[k] != 0)
	    phgPrintf(" %d:%"dFMT, k, cnts0[k]);
    phgPrintf("%s\n", cnts0[0] ? "" : " none");
    phgPrintf("  # global unknowns: %"dFMT, solver->nglobal);
    phgPrintf(" (%"dFMT" for the cut elements)\n", n0);
#endif	/* defined(PHG_TO_P4EST) */

    double np = 0.;
    for (k = 0; k < xi->nd; k++)
	np += phgQCTotalNP(qc[k], g->comm);
    assert(np <= ULONG_MAX);
    phgPrintf("  Total # quadrature points: %llu\n", (unsigned long)np);

    phgXFEMQCFree(xi, &qc);

    if (LDG) {
	INT i;
	process_mass_matrix(M);
	tmp = phgMatDiagonalBlockInverse(M);
#if 0
	{
	    MAT *tmp1;
	    /* check M^(-1) computed by phgMatDiagonalBlockInverse */
	    MAT *tmp1 = phgMatMat(MAT_OP_N, MAT_OP_N, 1.0, M, tmp, 0.0, NULL);
	    double Ioo = phgMatNormInfty(tmp1);
	    double Moo = phgMatNormInfty(M);
	    double invMoo = phgMatNormInfty(tmp);
	    phgPrintf("  inv(M) check: |M|oo = %0.2e, |M^-1|oo = %0.2e, "
					  "|M*M^-1|_oo = %0.2e\n",
					  Moo, invMoo, Ioo);

	    phgMatDestroy(&tmp1);
	}
#endif		
	phgMatDestroy(&M);

	/* add coefficient to M = C M^inv or M^inv C */
	M = phgMatMat(MAT_OP_N, MAT_OP_N, 1., C, tmp, 0., NULL);

	phgMatDestroy(&tmp);
	phgMatDestroy(&C);

	for (i = 0; i < Dim; i++) {
	    phgVecAssemble(F[i]);
	    /* matrix + (Lu, Lv): \sum_{d=1}^{Dim} B_d^{T} M^{-1} B_d
	       rhs + (L1(g))+L2(jD), Lv): \sum_{d=1}^{Dim} B_d^{T} M^{-1} F_d */

	    /* M = B_d^{T} M^{-1}  */
	    tmp = phgMatMat(MAT_OP_T, MAT_OP_N, 1.0, B[i], M, 0.0, NULL);
	    /* solver->mat +=  M B_d */
	    phgMatMat(MAT_OP_N, MAT_OP_N, 1.0, tmp, B[i], 1.0, &solver->mat);
	    /* solver->rhs +=  M F_d */
	    phgMatVec(MAT_OP_N, 1.0, tmp, F[i], 1.0, &solver->rhs);

	    phgMatDestroy(&tmp);
	    phgMatDestroy(&B[i]);
	    phgVecDestroy(&F[i]);
	}
	phgMatDestroy(&M);
    }

    /* set the diagonal entry of empty rows to 1 */
    phgXFEMProcessEmptyRows(solver);
}

#ifndef TEST_ORDER_FUNC
# define TEST_ORDER_FUNC 0
#endif

#if defined(PHG_TO_P4EST) && TEST_ORDER_FUNC
/* Sample user ORDER_FUNC function, to enble it:
 * 	make USER_CPPFLAGS="-DPHG_TO_P4EST -DTEST_ORDER_FUNC=1" */
static int
order_func(XFEM_INFO *xi, DOF_TYPE *type, ELEMENT *e, int pno, FLOAT eta)
{
    FLOAT h0 = 1.0 / (1<<xi->generation0);
    FLOAT h, corners[2][Dim];
    int p0 = type->order;

    if (e == NULL)
	return p0;	/* max order */

#if TEST_ORDER_FUNC == 2
# warning For debugging only!
    /* Note: use the anchor element for consistency within macro elements */
    e = Anchor(xi, xi->g, e, pno);
    return GlobalElement(xi->g, e->index) % type->order + 1;
#endif

    phgXFEMMacroCorners(xi, e->index, pno, corners);
    h = 0.;
    for (int k = 0; k < Dim; k++)
	if (h < corners[1][k] - corners[0][k])
	    h = corners[1][k] - corners[0][k];
    if (h >= h0 * 0.9)
	return p0;
    FLOAT theta = (1.+3.*eta)/(1.-eta);
    theta = theta + Sqrt(theta * theta - 1.0);
    /* Note: solve h0^p0 == (theta*h)^p for p */
    int p = ceil(p0*Log(h0) / Log(theta*h)) + 0.5;
    if (p > p0)
	p = p0;
    if (p <= 0)
	p = 1;
    return p;
}
#endif	/* defined(PHG_TO_P4EST) && TEST_ORDER_FUNC */

int
main(int argc, char *argv[])
{
    char *fn = "../mesh/cube.mesh", *fn_CAD = NULL;
    int i, level, total_level = 0;
    INT refine = 0, npart = 0, nd;
    BOOLEAN interior = TRUE;
#ifdef PHG_TO_P4EST
    INT refine0 = 1, refine_step = 1;
#else	/* defined(PHG_TO_P4EST) */
    INT refine0 = 0, refine_step = 3;
#endif	/* defined(PHG_TO_P4EST) */
    GRID *g;
    XFEM_INFO *xi;
    QI_CTX *qic = NULL;
    DOF **u_h, **f_h, **gu_h, **err, **gerr;
    DOF **u_old, *ls = NULL, *ls_grad = NULL;
    SOLVER *solver, *pc;
    size_t mem_peak;
    double t;
    INT corner_flags = 0;
    FLOAT L2norm, H1norm, L2err, H1err, d;
    BOOLEAN vtk = FALSE, debug_pre = FALSE;
    ELEMENT *e;

    phgOptionsRegisterFilename("-mesh_file", "Mesh file", &fn);
#if HAVE_OPENCASCADE
    phgOptionsRegisterFilename("-CAD_file", "CAD model file", &fn_CAD);
#endif	/* HAVE_OPENCASCADE */
    phgOptionsRegisterNoArg("-LDG", "Use local DG instead of IPDG", &LDG);
    phgOptionsRegisterInt("-refine0", "Initial refinement levels", &refine0);
    phgOptionsRegisterInt("-refine", "Repeated refinement levels", &refine);
    phgOptionsRegisterInt("-refine_step", "Refinement step", &refine_step);
    phgOptionsRegisterFloat("-c",  "Radius or offset of the interface", &c);
    phgOptionsRegisterFloat("-xc", "Center or normal of the interface: x", &xc);
    phgOptionsRegisterFloat("-yc", "Center or normal of the interface: y", &yc);
    phgOptionsRegisterFloat("-zc", "Center or normal of the interface: z", &zc);
    phgOptionsRegisterFloat("-coeff1", "The coefficient coeff1", &coeff[0]);
    phgOptionsRegisterFloat("-coeff2", "The coefficient coeff2", &coeff[1]);
    phgOptionsRegisterFloat("-beta", "The parameter beta", &beta);
    phgOptionsRegisterFloat("-gamma0", "The parameter gamma0", &gamma0);
    phgOptionsRegisterFloat("-gamma1", "The parameter gamma1", &gamma1);
    phgOptionsRegisterFloat("-gamma2", "The parameter gamma2", &gamma2);
    phgOptionsRegisterFloat("-gammat", "The parameter gammat", &gammat);
    phgOptionsRegisterInt("-added_order", "Extra quadrature orders for "
					"interface elements", &added_order);
    phgOptionsRegisterNoArg("-vtk", "Create \"unfitted.vtk\"", &vtk);

    phgOptionsRegisterNoArg("-dump_solver", "Output matrix/RHS as .m files",
					&dump_solver);
    phgOptionsRegisterNoArg("-dump_reorder", "Reorder unknowns in the .m files",
					&dump_reorder);

    phgOptionsRegisterInt("-ls_order", "Polynomial order of the levelset"
			    "function (1: plane, 2: sphere)", &ls_order);
    phgOptionsRegisterInt("-sol_order", "Analytic solution's polynomial order "
		"(<0: non-poly., >0 => -no_jump, -99 => FE order)", &sol_order);
    phgOptionsRegisterNoArg("-no_jump", "No jump in the solution across the "
			    "surfaces", &no_jump);
    phgOptionsRegisterInt("-corner_flags", "Only refine corner elements during "
			  "initial refinements ((corner_flags&(1<<k))!=0 => "
			  "refine elements at corner k)", &corner_flags);
    phgOptionsRegisterNoArg("-debug_pre", "Set initial x0 to analytic solution",
			    &debug_pre);
    phgOptionsRegisterInt("-npart", "Maximum number of parts", &npart);
    phgOptionsRegisterNoArg("-interior_only", "Only include the interior parts "
			    "(immersed boundary problem)", &interior);
		phgOptionsRegisterInt("-gp_type", "Ghost penalty type(0.Face_based, 1.Element_based)",
					 &gp_type);

		phgOptionsPreset("-xfem_ctol=0 -xfem_etol=0 -xfem_ortho=none");
    // phgOptionsPreset("-solver_scaling=sym");
		// phgOptionsPreset("-solver mumps");
    phgOptionsPreset("-dof_type Q1");
		phgOptionsRegisterNoArg("-use_pc", "Use MG preconditioned", &use_pc);


    phgInit(&argc, &argv);

    if (sol_order == -99)
	sol_order = DOF_DEFAULT->order;
    else if (sol_order == -100)
	interior = TRUE;

    g = phgNewGrid(-1);
    if (!phgImport(g, fn, TRUE))
	phgError(1, "can't read file \"%s\".\n", fn);

    if (fn_CAD == NULL)
	ls_info(g);

    if (LDG) {
	const char *xfem_dof_type = phgOptionsGetString("-xfem_dof_type");
	if (xfem_dof_type != NULL && strcmp(DOF_DEFAULT->name, xfem_dof_type))
	    phgError(1, "local DG requires -dof_type == -xfem_dof_type\n");
	if (DOF_DEFAULT->fe_space != FE_L2)
	    phgError(1, "local DG requires a DG element\n");
	beta = 1.;
    }

    /* refine the mesh */
    phgPrintf("Initial refinement (refine0=%d): ", refine0);
    while (refine0 > 0) {
	BOOLEAN marked = FALSE;
	level = refine0 > refine_step ? refine_step : refine0;
	if (corner_flags) {
	    /* only refine the corner elements (for debugging) */
	    ELEMENT *e;
	    int k, ncorners = (1 << Dim);
	    ForAllElements(g, e) {
		e->mark = 0;
		for (k = 0; k < ncorners; k++) {
		    if (!(corner_flags & (1 << k)))
			continue;
		    for (i = 0; i < NVert; i++)
#ifdef PHG_TO_P4EST
			if (e->corners[(i>>0)&1][0] == g->bbox[(k>>0)&1][0] &&
			    e->corners[(i>>1)&1][1] == g->bbox[(k>>1)&1][1] &&
			    e->corners[(i>>2)&1][2] == g->bbox[(k>>2)&1][2])
			    break;
#else	/* defined(PHG_TO_P4EST) */
			if (g->verts[e->verts[i]][0] == g->bbox[(k>>0)&1][0] &&
			    g->verts[e->verts[i]][1] == g->bbox[(k>>1)&1][1] &&
			    g->verts[e->verts[i]][2] == g->bbox[(k>>2)&1][2])
			    break;
#endif	/* defined(PHG_TO_P4EST) */
		    if (i < NVert)
			break;
		}
		if (k >= ncorners)
		    continue;
		e->mark = level;
		marked = TRUE;
	    }
	}
	if (marked)
	    phgRefineMarkedElements(g);
	else
	    phgRefineAllElements(g, level);	/* uniform refinement */
    	phgBalanceGrid(g, 1.1, 1, NULL, 1.0);
	refine0 -= level;
	total_level += level;
    }
    phgPrintf("%"dFMT" element%s.\n", g->nelem_global,
					g->nelem_global > 1 ? "s" : "");

    if (fn_CAD == NULL) {
	/* create DOFs for the level set function */
#if 1
	/* use analytic function */
	ls = phgDofNew(g, DOF_ANALYTIC, 1, "ls", ls_func);
	/*phgDofSetPolyOrder(ls, ls_order);*/
	ls_grad = phgDofNew(g, DOF_ANALYTIC, 3, "ls_grad", ls_grad_func);
	/*phgDofSetPolyOrder(ls_grad, ls_order <= 0 ? ls_order : ls_order-1);*/
#else
	/* project the levelset function to a FE space */
	assert(ls_order >= 1);
	ls = phgDofNew(g, DOF_DGn[ls_order], 1, "ls", ls_func);
	ls_grad = phgDofNew(g, DOF_DGn[ls_order-1], 3, "ls_grad", ls_grad_func);
	/*ls->userfunc = DofInterpolation;
	ls_grad->userfunc = DofInterpolation;*/
#endif
	npart = 1;
	nd = interior ? 1 : 2;
    }
    else {
	while (isspace(*fn_CAD))
	    fn_CAD++;
	i = strlen(fn_CAD);
	while (i > 0 && isspace(fn_CAD[i - 1]))
	    i--;
	fn_CAD[i] = '\0';
	qic = phgOCCInitQIC(g, !interior, npart, NULL,
				fn_CAD[0] == '&' ? fn_CAD + 1 : fn_CAD);
	if (qic == NULL)
	    phgError(1, "Error in loading CAD file, abort.\n");
	npart = qic->npart;
	nd = qic->nd;
	if (fn_CAD[0] == '&')
	    exit(0);
    }

    u_h = phgXFEMDofNew0(nd, g, DOF_DEFAULT, 1, "u_h", DofInterpolation);

    t = phgGetTime(NULL);
    while (TRUE) {
	/* variables for statistics */
	double h_min = 1e10, h_max = 0.0, nnz, nnz_d, nnz_o;
	phgPrintf("\n********** Level %d, %d proc%s, %"
		  dFMT" elem%s, LIF %0.2lf, refine time: %0.4lg\n",
		  total_level, g->nprocs, g->nprocs > 1 ? "s" : "",
		  g->nleaf_global, g->nleaf_global > 1 ? "s" : "",
		  (double)g->lif, phgGetTime(NULL) - t);

#if defined(PHG_TO_P4EST) && TEST_ORDER_FUNC
	phgXFEMSetOrderFunc(order_func);
#endif	/* defined(PHG_TO_P4EST) && TEST_ORDER_FUNC */

	phgPrintf("Resolving/merging interface elements: ");
	t = phgGetTime(NULL);
	if (fn_CAD == NULL)
	    /* use level set function */
	    xi = phgXFEMInitLS(ls, ls_grad, ls_order, !interior,
					2 * DOF_DEFAULT->order,
					2 * DOF_DEFAULT->order + added_order);
	else
	    /* use CAD model */
	    xi = phgXFEMInit(qic, g, 2 * DOF_DEFAULT->order,
					2 * DOF_DEFAULT->order + added_order);
#if 0
# warning Testing phgXFEMInit
phgXFEMFree(&xi);
break;
#endif
	phgXFEMInfo(xi, stdout, "  ");

	/* get h_max and h_min (for information only) */
	ForAllElements(g, e) {
#ifdef PHG_TO_P4EST
	    FLOAT (*corners)[Dim] = phgGeomGetCorners(g, e, NULL);
	    d = 0.;
	    for (i = 0; i < Dim; i++) {
		FLOAT e = corners[1][i] - corners[0][i];
		if (d < e)
		    d = e;
	    }
#else	/* defined(PHG_TO_P4EST) */
	    d = 0.;
	    for (i = 0; i < NEdge; i++) {
		COORD *c0 = g->verts + e->verts[GetEdgeVertex(i,0)];
		COORD *c1 = g->verts + e->verts[GetEdgeVertex(i,1)];
		FLOAT v[] = {(*c0)[0] - (*c1)[0], (*c0)[1] - (*c1)[1],
			     (*c0)[2] - (*c1)[2]};
		FLOAT e = Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (d < e)
		    d = e;
	    }
#endif	/* defined(PHG_TO_P4EST) */
	    if (h_max < d)
		h_max = d;
	    if (h_min > d)
		h_min = d;
	}
#if USE_MPI
	if (g->nprocs > 1) {
	    double tmp0[2] = {-h_min, h_max}, tmp[2];
	    MPI_Reduce(tmp0, tmp, 2, MPI_DOUBLE, MPI_MAX, 0, g->comm);
	    h_min = -tmp[0];
	    h_max = tmp[1];
	    if (g->rank > 0)
		h_min = h_max = 1.0;
	}
#endif	/* USE_MPI */

	phgPrintf("Building linear equations (%"dFMT" elems, FE: %s/%s) ...\n"
		  "  Element sizes: max/min = %0.2le/%0.2le = %d\n",
			g->nelem_global, u_h[0]->type->name,
			phgXFEMDofTypeName(xi,u_h[0],NULL,0),
			h_max, h_min, (int)(h_max/h_min+0.5));

	phgSetupHalo(g, HALO_FACE);	/* HALO_FACE is needed for DG */

	if (beta == 1.0) {
	    if (!phgOptionsIfUsed("-solver_symmetry"))
		phgOptionsSetOptions("-solver_symmetry=sym");
	    if (!phgOptionsIfUsed("-solver") && FT_PHG == FT_DOUBLE)
		phgOptionsSetOptions("-solver minres");
	}
	else if (!phgOptionsIfUsed("-solver"))
	    phgOptionsSetOptions("-solver gmres");
	solver = phgXFEMSolverCreate(xi, SOLVER_DEFAULT, 0, u_h, NULL);
	solver->mat->handle_bdry_eqns = FALSE;
	solver->rtol = solver->btol = 1e-12;
	if (use_pc) {
      phgSolverXASPSetMG(pc, g, u_h[0]->type->order);
      phgSolverXASPSetXFEMINFO(pc, xi); 
      pc = phgMat2Solver(SOLVER_XASP, solver->mat);
      pc->maxit = 1;
      pc->monitor = FALSE;
      pc->warn_maxit = FALSE;
      phgSolverSetPC(solver, pc, NULL);
	}

	t = phgGetTime(NULL);
	f_h = phgXFEMDofNew(xi, DOF_DEFAULT, 1, "f1_h", DofNoAction);
	phgXFEMDofSetDataByFunction(xi, f_h, func_f);
	build_linear_system(xi, solver, u_h, f_h);
	phgXFEMDofFree(xi, f_h);

	nnz_d = solver->mat->nnz_d;
	nnz_o = solver->mat->nnz_o;
	nnz = nnz_d + nnz_o;
#if USE_MPI
	if (g->nprocs > 1) {
	    double tmp0[2] = {nnz_d, nnz_o}, tmp[2];
	    MPI_Reduce(tmp0, tmp, 2, MPI_DOUBLE, MPI_MAX, 0, g->comm);
	    nnz_d = tmp[0];
	    nnz_o = tmp[1];
	    tmp[0] = nnz;
	    MPI_Reduce(tmp, &nnz, 1, MPI_DOUBLE, MPI_SUM, 0, g->comm);
	}
#endif	/* USE_MPI */
	phgPrintf("  # local nonzeros: max inproc/offproc %0.0lf/%0.0lf, "
		  "total %0.0lf\n", nnz_d, nnz_o, nnz);

	phgMemoryUsage(g, &mem_peak);
	phgPrintf("  Wall time: %0.2lgs, memory usage: %0.2lfGB\n",
		  phgGetTime(NULL)-t, (double)mem_peak/(1024.*1024.*1024.));

	phgPrintf("Solving linear equations ...\n");
	t = phgGetTime(NULL);
	u_old = phgXFEMDofCopy(xi, u_h, NULL, NULL, "u_old");    

	if (debug_pre)
	    /* debug to_DG/pre_solver with "-sol_order=-99 -added_order=15",
	     * # of iterations should \approx 1 */
	    phgXFEMDofSetDataByFunction(xi, u_h, func_u);

	/* Note: phgMapE2G cannot be used after solver->mat is assembled */
	if (dump_solver)
	    phgSolverDumpMATLAB_(solver, "A", "b", "p", dump_reorder);

	phgXFEMSolve(xi, solver, TRUE, u_h, NULL);

	phgPrintf("  solver %s; nits: %d; residual: %lg; time: %0.4lg\n",
	    solver->oem_solver->name, solver->nits, (double)solver->residual,
	    phgGetTime(NULL) - t);
	if (solver->cond > 0.0)
	    phgPrintf("  Condition number: %0.2le\n", (double)solver->cond);
	phgSolverDestroy(&solver);
	if (use_pc)
	  phgSolverDestroy(&pc);

	/* compute L2 and H1 errors */
	t = phgGetTime(NULL);

	err = phgXFEMDofNew(xi, DOF_DEFAULT, 1, "error", DofNoAction);
	phgXFEMDofSetDataByFunction(xi, err, func_u);
	gu_h = phgXFEMDofGradient(xi, u_h, NULL, NULL, NULL);
	gerr = phgXFEMDofGradient(xi, err, NULL, NULL, NULL);

	/* norms of the analytic solution */
	H1norm = L2norm = Sqrt(phgXFEMDofDot(xi, err, err));
	H1norm += Sqrt(phgXFEMDofDot(xi, gerr, gerr));

	/* adjust norms by the numerical solution */
	d = Sqrt(phgXFEMDofDot(xi, u_h, u_h));
	if (L2norm < d)
	    L2norm = d;
	d = L2norm + Sqrt(phgXFEMDofDot(xi, gu_h, gu_h));
	if (H1norm < d)
	    H1norm = d;

	/* norms of the numerical error */
	phgXFEMDofAXPY(xi, -1.0, u_h, &err);
	H1err = L2err = Sqrt(Fabs(phgXFEMDofDot(xi, err, err)));
	phgXFEMDofAXPY(xi, -1.0, gu_h, &gerr);
	phgXFEMDofFree(xi, gu_h);
	H1err += Sqrt(Fabs(phgXFEMDofDot(xi, gerr, gerr)));
	phgXFEMDofFree(xi, gerr);

	phgMemoryUsage(g, &mem_peak);
	phgPrintf("L2err: %0.10le; H1err: %0.10le; "
		  "mem: %0.2lfGB; time: %0.2lg\n",
		    (double)L2err / (L2norm == 0. ? 1.0 : (double)L2norm),
		    (double)H1err / (H1norm == 0. ? 1.0 : (double)H1norm),
		    (double)mem_peak / (1024. * 1024. * 1024.),
		    phgGetTime(NULL) - t);

	phgXFEMDofAXPY(xi, -1.0, u_h, &u_old);
	phgPrintf("  |u_h-u_H| = %0.5le\n",
			(double)Sqrt(Fabs(phgXFEMDofDot(xi, u_old, u_old))));

	if (vtk) {
	    char name[128];
	    const char *ret;
	    sprintf(name, "unfitted-%02d.vtk", total_level);
	    ret = phgXFEMExportVTK(g, name, u_h, err, NULL);
	    phgPrintf("\"%s\" created.\n", ret);
	}

	phgXFEMDofFree(xi, u_old);
	phgXFEMDofFree(xi, err);
	phgXFEMFree(&xi);

	if (refine <= 0)
	    break;

	level = refine > refine_step ? refine_step : refine;
	phgRefineAllElements(g, level);
	phgBalanceGrid(g, 1.2, 0, NULL, 1.0);
	refine -= level;
	total_level += level;
    }

    phgQIFree(&qic);

    phgXFEMDofFree0(nd, u_h);
    phgDofFree(&ls);
    phgDofFree(&ls_grad);
    phgFreeGrid(&g);

    phgFinalize();

    return 0;
}
