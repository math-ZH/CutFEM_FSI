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

/* XASP, Auxiliary Subspace Preconditioner, solves the following
 * interface problem using XFEM:
 *
 *	-\div (alpha \grad u) + beta u = f,		in \Omega^- \cup \Omega^+,
 *			                         u = g_D,	on \partial\Omega_D,
 *		               alpha \grad u = g_N \cdot n,	on \partial\Omega_N,
 *		                         [u] = j_D,	on \Gamma,
 *	                    [a\grad u] = j_N \cdot n,	on \Gamma,
 * Where
 * 	\Omega^- := \Omega \cap \{x: L(x)<0\},
 * 	\Omega^+ := \Omega \cap \{x: L(x)>0\},
 * 	\Gamma	 := \Omega \cap \{x: L(x)=0\},
 * alpha/beta := func1(x) for x \in \Omega^- and func2(x) for x \in \Omega^+
 * alpha > 0, beta >= 0 for x \in \Omega.
 * L(x) is the level-set function defined by ls_func().
 * 
 * $Id: xasp.h ,v 1.9 2025/04/25 zdh $
 */
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdlib.h>

#include "phg.h"
#include "phg/xfem.h" 

/* Parameters for MG */                        
/* This data structure is shared by all the levels */                           
typedef struct {
    INT mg_type;              /* type of MG: 1 non-full 2 full */
    INT cycle_type;           /* type of MG cycle : 1 V 2 W */
    INT	coarse_matrix_type;   /* type of coarse_matrix :  */
                              /* 1 (P^tA)P 2 P^t(AP) */
    INT finest_order;         /* order of finest scale */
    INT coarsest_order;       /* order of coarsest scale */
    INT presmooth_iter;       /* number of presmoothers */
    INT postsmooth_iter;      /* number of postsmoothers */
    FLOAT damped;             /* damping parameters for jacobi */
} MG_PARAM; 

 /* Data for MG */                         
 /* Each level has its own copy of this data */
typedef struct {
    MAT *A;      // pointer to the matrix at level level_num
    MAT *P;      // prolongation operator at level level_num
    VEC *b;      // pointer to the right-hand side at level level_num
    VEC *x;      // pointer to the iterative solution at level level_num
    INT order;   // polynoial's highest order
} MG_DATA; 
                                                       
/* Data for ASP */
typedef struct {
    XFEM_INFO *xi;                  /* XFEM information structure xi */
    GRID  *g;                       /* GRID object */
    MAT   *Pc, *Pi;                 /* The transfer matrices Pc and Pi */
    MAT		*Ac, *Ai;	                /* The stiffness matrices with V^c and V^i */
    SOLVER *aux_solver;             /* Auxiliary solver for V^i */
    MG_DATA *mg_data;               /* Data for MG */ 
    SOLVER  *mg_approx_solver;      /* Coarsest approx solver for MG */

    char	*cycle_type;              /* XASP cycle type */
    char	*aux_solver_opts;         /* Solver options for auxiliary solver */
    char  *mg_approx_solver_opts;   /* Solver options for approx solver of MG */
    MG_PARAM mg_param;              /* Paramerts for MG */
    BOOLEAN dump_mat;               /* whether dump auxiliary matrix */
    BOOLEAN	assembled;              
} OEM_DATA;

/* global parameters */ 
static char *cycle_type = "12";	    /* default cycle type */
static char	*aux_solver_opts	= NULL;
static MG_PARAM mg_global_params={
    /* mg_type */ 1,
    /* cycle_type: W-cycle */ 2,
    /* coarse_matrix_type: (P^t)AP */ 1,
    /* finest_order */ 2, 
    /* coarsest_order*/ 1, 
    /* presmooth_iter */ 2, 
    /* postsmooth_iter */ 2,
    /* damped */ 1
};
static char *mg_approx_solver_opts = NULL;
static BOOLEAN dump_mat = FALSE;

/* convenience macro */
#define _t	((OEM_DATA *)solver->oem_data)
#define _mg_data	(_t->mg_data)
#define _mg_param (_t->mg_param)

#define Initialize		NULL
#define Finalize		NULL
#define AddMatrixEntries	NULL
#define AddRHSEntries		NULL
#define SetPC			NULL

/*---------------------------- export functions -------------------------*/
void
phgSolverXASPSetMG(SOLVER *solver, GRID *g, INT order)
{
    if (_t->assembled){
        phgWarning("phgSolverXASPSetMG() should be \
                 called before phgSolverAssemble()!\n");
        return;
    }

    if (_t->g != NULL)
  phgFreeGrid(&_t->g);

    if (g != NULL)
  _t->g = g;

  mg_global_params.finest_order = order;

    return;
}

void
phgSolverXASPSetXFEMINFO(SOLVER *solver, XFEM_INFO *xi)
/* defines the xfem object in solver */
{
    if (_t->assembled){
        phgWarning("phgSolverXASPSetXFEMINFO() should be \
                 called before phgSolverAssemble()!\n");
        return;
    }

    assert(xi != NULL);

    if (_t->xi != NULL)
  phgXFEMFree(&_t->xi);

    if (xi != NULL) 
  _t->xi = xi;
   
    if (_t->Ac != NULL)
        phgMatDestroy(&_t->Ac);
    if (_t->Ai != NULL)
        phgMatDestroy(&_t->Ai);

    return ;
}

void
phgSolverXASPSetSubmatrix(SOLVER *solver, MAT *Ac, MAT *Ai)
{
   if (_t->assembled){
        phgWarning("phgSolverXASPSetSubmatrix() should be \
                        called before phgSolverAssemble()!\n");
        return ;
   }

   assert(Ac != NULL);

   if (_t->Ac != NULL)
        phgMatDestroy(&_t->Ac);
   if (_t->Ai != NULL)
       phgMatDestroy(&_t->Ai);

   Ac->refcount++;
   _t->Ac = Ac;
   Ai->refcount++;
   _t->Ai = Ai;

   return;
}

/*--------------------------------------------------------------------------*/

/*---------------------------- auxiliary functions -------------------------*/
#define MAX_(d, length, max) { \
  max = -1; \
  for (int i = 0; i < length; i++) \
      if (d[i] > max) \
    max = d[i];}
#define SUM_(d, length, sum) { \
  sum = 0; \
  for (int i = 0; i < length; i++) \
      sum += d[i];}

static BOOLEAN 
judge_order(DOF *u, ELEMENT *e,  int index, int order)
{
    DOF_TYPE *type = u->type;
    int alpha[Dim], a, m, p = DofTypeOrder(u, e);
    BOOLEAN judge = FALSE;

    a = type->orders[index];
    alpha[0] = a % (p + 1);
    for (m = 1; m < Dim; m++) {
    a /= p + 1;
    alpha[m] = (m == Dim -1) ? a : a % (p + 1);
    }
    if (Dim == 3) {
      int max;
      MAX_(alpha, Dim, max);
      if (max <= order) {
       judge = TRUE;
      }
    }
    else {
      int sum; 
      SUM_(alpha, Dim, sum);
      if (sum <= order) {
       judge = TRUE;
      }
    }

    return judge;
}

#if 0
/* debugging a given Matrix entry */
# define DebugMat \
    phgInfo(-1, "%s:%d: P(%"dFMT",%"dFMT") set 1\n", \
		__func__, __LINE__, I+1, J+1);
#else
#define DebugMat	/* do nothing */
#endif

/* prolongation matrix */ 
static MAT *
build_MG_P(XFEM_INFO *xi, SOLVER *solver, INT level)
{
    GRID *g = _t->g;
    INT order_f = _mg_data[level].order;
    INT order_c = _mg_data[level+1].order;
    ELEMENT *e;
    DOF *u_f, *u_c;
    MAP *map_f, *map_c;
    MAT *P = NULL;

	  u_f = phgDofNew(g, DOF_DGPn[order_f], 1, "u_f", DofInterpolation);
	  u_c = phgDofNew(g, DOF_DGPn[order_c], 1, "u_c", DofInterpolation);
    map_f = phgMapCreate(u_f, NULL);
    map_c = phgMapCreate(u_c, NULL);
    P = phgMapCreateMat(map_f, map_c);
    P->handle_bdry_eqns=FALSE;

    int nd = xi->nd;
    ForAllElementsBegin(g, e) {
	int N_f = DofNBas(u_f, e);
	int N_c = DofNBas(u_c, e);
  int is_c_local_indices[N_c];
	INT I, J;
	int i, j;

	/* loop on the elements */
  int l = 0;
  for (i = 0; i < N_f; i++) {
    if (judge_order(u_f, e, i, order_c)) {
        is_c_local_indices[l] = i;
        l++;
    }
  }
	for (i = 0; i < N_f; i++) {
      xi->nd = 1;
      /* get global index */
	    I = phgMapE2G(P->rmap, 0, e, i);
	    for (j = 0; j < N_c; j++) {
    /* get global index */
		J = phgMapE2G(P->cmap, 0, e, j);
    /* judge order */ 
    if (i == is_c_local_indices[j])
    /* assemble to global matrix */
		  phgMatAddGlobalEntry(P, I, J, 1.0); 
      DebugMat
    }
      xi->nd = nd;
	}

    } ForAllElementsEnd

    phgDofFree(&u_f);
    phgDofFree(&u_c);

	  phgMatAssemble(P);
    return P;
}

static void 
phgMATProcessEmptyRows(MAT *mat)
/* process empty row */
{
  int i;
  MAT_ROW *row;

  phgMatDisassemble(mat);
  assert(mat->type == PHG_UNPACKED);
  for (i = 0; i < mat->rmap->nlocal; i++) {
    row = mat->rows + i;
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
  phgMatAssemble(mat);

  return;
}

/* builds the coarse grid matrix.
 *	_mg_param.coarse_matrix_type == 1: use (P^tA)P
 *	_mg_param.coarse_matrix_type == 2: use P^t(AP)
 **/
static MAT *
build_MG_C(SOLVER *solver, INT level)
{
  MAT *C = NULL,  *T;
  assert(_mg_data[level].A != NULL);

	if (_mg_data[level+1].P->rmap->nglobal < _mg_data[level+1].P->cmap->nglobal)
	phgError(1, "%s (%s:%d): more DOF on coarse grid than on "
      "fine grid!\n", __func__, __FILE__, __LINE__);

  switch (_mg_param.coarse_matrix_type) {
	case 1:		/* C = (P^tA)P */
	    T = phgMatMat(MAT_OP_T, MAT_OP_N, 1., _mg_data[level+1].P, 
          _mg_data[level].A, 0., NULL);
	    C = phgMatMat(MAT_OP_N, MAT_OP_N, 1., T, _mg_data[level+1].P, 0., NULL);
	    phgMatDestroy(&T);
	    break;
	case 2:		/* C = P^t(AP) */
	    T = phgMatMat(MAT_OP_N, MAT_OP_N, 1., _mg_data[level].A, 
          _mg_data[level+1].P, 0., NULL);
	    C = phgMatMat(MAT_OP_T, MAT_OP_N, 1., _mg_data[level+1].P, T, 0., NULL);
	    phgMatDestroy(&T);
	    break;
	default:
	    phgError(1, "%s:%d: invalid coarse matrix type (%d).\n",
			__FILE__, __LINE__, _mg_param.coarse_matrix_type);
    }
  
  phgMATProcessEmptyRows(C);

    return C;
}

/* damped jacobi iteration function */
static void 
damped_chebyshev(SOLVER *solver, MAT *A, VEC *b, VEC **px, FLOAT rho, int max_its)
{
    VEC *x, *r, *w;
    INT i, nits;
    FLOAT b_norm;

    if (max_its <= 0)
      return;

    if (*px == NULL || b == NULL)
      phgError(1, "%s:%d invalid vector handler\n", __FILE__, __LINE__);

    assert(b->nvec == 1);

    x = *px;
    r = phgMapCreateVec(b->map, 1);
    w = phgMapCreateVec(b->map, 1);
    bzero(w->data, sizeof(*w->data) * w->map->nlocal);
    if (IsZero(b_norm = phgVecNormInfty(b, 0, NULL))){
      bzero(x->data, sizeof(*x->data) * x->map->nlocal);
      return;
    }

    if (A->diag == NULL)
      phgMatSetupDiagonal(A);

    nits = 0;
    while (TRUE) {
      phgVecCopy(b, &r);
      phgMatVec(MAT_OP_N, -1.0, A, x, 1.0, &r);
      for (i = 0; i < A->cmap->nlocal; i++) {
        w->data[i] = (2 * nits - 1) / (2 * nits + 3 ) * w->data[i];
        w->data[i] += (8 * nits + 4) / (2 * nits + 3 ) * 
          (r->data[i] / A->diag[i]) / (rho * _mg_param.damped);
        x->data[i] += w->data[i];
      }
      if (++nits > max_its)
        break;
    }          
    phgVecDestroy(&r);
    phgVecDestroy(&w);

}

/* non-recursive multigrid cycle function */
static void 
nrmg(SOLVER *solver)
{
    INT nl = _mg_param.finest_order - _mg_param.coarsest_order + 1;
    INT num_presmooth[10] = {0}, l = 0;
    FLOAT rho = (FLOAT)_mg_param.finest_order + 1; /* rho = p + 1; */
    VEC *r0 = NULL;
ForwardSweep:
    while (l < nl-1) {
      num_presmooth[l]++;
      /* pre-smoothing */ 
      damped_chebyshev(solver, _mg_data[l].A, _mg_data[l].b, &(_mg_data[l].x), 
          rho, _mg_param.presmooth_iter);
      /* residual r0 = b - A * x */
      if (r0 == NULL)
          r0 = phgMapCreateVec(_mg_data[l].A->rmap,1);
      phgVecCopy(_mg_data[l].b, &r0);
		  phgMatVec(MAT_OP_N, -1.0, _mg_data[l].A, _mg_data[l].x, 1.0, &r0);
      /* restriction r1 = R * r0 */
      phgMatVec(MAT_OP_T, 1.0, _mg_data[l+1].P, r0, 0.0, &(_mg_data[l+1].b));
      /* prepare for the next level */
      ++l;
      bzero(_mg_data[l].x->data, 
          _mg_data[l].x->map->nlocal * sizeof(*(_mg_data[l].x)->data));
      phgVecDestroy(&r0);
    }

    /* call the approximation space solver: */
    _t->mg_approx_solver->rhs = _mg_data[nl-1].b;
	  _t->mg_approx_solver->rhs->assembled = TRUE;
	  _t->mg_approx_solver->rhs_updated = TRUE;
	  _t->mg_approx_solver->rhs->mat = _mg_data[nl-1].A;
	  phgSolverVecSolve(_t->mg_approx_solver, FALSE, _mg_data[nl-1].x);
	  _t->mg_approx_solver->rhs = NULL;

    /* BackwardSweep: */
    while (l > 0) {
      --l;
      /* prolongation u = u + P * e1 */
      phgMatVec(MAT_OP_N, 1.0, _mg_data[l+1].P, 
          _mg_data[l+1].x, 1.0, &(_mg_data[l].x));
      /* post-smoothing */ 
      damped_chebyshev(solver, _mg_data[l].A, _mg_data[l].b, &(_mg_data[l].x), 
          rho, _mg_data[l].order);
      /* _mg_param.cycle_type: 1 V cycle, 2 W cycle */ 
      if (num_presmooth[l] < _mg_param.cycle_type) 
        break;
      else 
        num_presmooth[l] = 0;
    }

    /* GOTO ForwardSweep */ 
    if (l > 0) goto ForwardSweep;

    return;
}

/* recursive multigrid cycle function*/
static void 
rmg(SOLVER *solver, INT l)
{ 
    INT nl = _mg_param.finest_order - _mg_param.coarsest_order + 1;
    INT i ;
    FLOAT rho = (FLOAT)_mg_param.finest_order + 1; /* rho = p + 1; */
    VEC *r0 = NULL; // fine l residual
    if (l < nl - 1) {
      /* pre smoothing */
      damped_chebyshev(solver, _mg_data[l].A, _mg_data[l].b, &(_mg_data[l].x), 
          rho, _mg_data[l].order);
      /* form residual r = b - A x */
      if (r0 == NULL)
          r0 = phgMapCreateVec(_mg_data[l].A->rmap,1);
      phgVecCopy(_mg_data[l].b, &r0);
      phgMatVec(MAT_OP_N, -1.0, _mg_data[l].A, _mg_data[l].x, 1.0, &r0);
      /* restriction r1 = R*r0 */
      phgMatVec(MAT_OP_T, 1.0, _mg_data[l+1].P, r0, 0.0, &(_mg_data[l+1].b)); 
      phgVecDestroy(&r0);
      /* call MG recursively: type = 1 for V cycle, type = 2 for W cycle */
      bzero(_mg_data[l+1].x->data, 
          _mg_data[l+1].x->map->nlocal * sizeof(*(_mg_data[l+1].x)->data));
      for (i = 0; i< _mg_param.cycle_type; ++i) 
        rmg(solver, l+1);
      /* prolongation u = u + P*e1 */
      phgMatVec(MAT_OP_N, 1.0, _mg_data[l+1].P, 
          _mg_data[l+1].x, 1.0, &(_mg_data[l].x));
      /* post smoothing */
      damped_chebyshev(solver, _mg_data[l].A, _mg_data[l].b, &(_mg_data[l].x), 
          rho, _mg_data[l].order);
    }
    else { 
      _t->mg_approx_solver->rhs = _mg_data[l].b;
      _t->mg_approx_solver->rhs->assembled = TRUE;
      _t->mg_approx_solver->rhs_updated = TRUE;
      _t->mg_approx_solver->rhs->mat = _mg_data[l].A;
      phgSolverVecSolve(_t->mg_approx_solver, FALSE, _mg_data[l].x);
           _t->mg_approx_solver->rhs = NULL;  
    }
}

/* full multigrid cycle function */
static void 
fmg (SOLVER *solver)
{
    INT nl = _mg_data[0].order - _mg_param.coarsest_order + 1;
    INT l = 0;
    FLOAT rho = (FLOAT)_mg_param.finest_order + 1; /* rho = p + 1; */
    VEC *r0 = NULL;
    /* restrict the RHS through all levels to coarsest. */
    for (l = 0; l < nl - 1; l++){
    	/* pre smoothing */ 
      damped_chebyshev(solver, _mg_data[l].A, _mg_data[l].b, &(_mg_data[l].x), 
          rho, _mg_data[l].order);
    	/* form residual r = b - A x */
      if (r0 == NULL)
          r0 = phgMapCreateVec(_mg_data[l].A->rmap,1);
      phgVecCopy(_mg_data[l].b, &r0);
      phgMatVec(MAT_OP_N, -1.0, _mg_data[l].A, _mg_data[l].x, 1.0, &r0);
      phgMatVec(MAT_OP_T, 1.0, _mg_data[l+1].P, r0, 0.0, &(_mg_data[l+1].b)); 
      phgVecDestroy(&r0);
    }
    /* work our way up through the levels */
    bzero(_mg_data[l].x->data, 
        _mg_data[l].x->map->nlocal * sizeof(*(_mg_data[l].x)->data));
    for (l = nl - 1; l > 0; l--) {
        rmg(solver, l);
        /* prolongation u = u + P*e1 */
        phgMatVec(MAT_OP_N, 1.0, _mg_data[l].P, 
            _mg_data[l].x, 1.0, &(_mg_data[l-1].x));
    }
    rmg(solver, l);
    return;
}


static void 
build_ASP_P(XFEM_INFO *xi, SOLVER *solver)
/* Builds the transfer matrices Pc (Vh->Vc) and Pi (Vh->Vi).
 * We denote by {\psi} and {\phi}, respectively, the finite element bases of
 * the Vh and Vc/Vi spaces, then the elements of the transfer matrix P
 * is determined by representing \phi with \psi:
 *		\psi_j = \sum_k P_{k,j} \phi_k 
 */
{
    MAP *map = solver->mat->rmap, *map_c;
    GRID *g = xi->g;
    ELEMENT *e;
    DOF *dof_c;

    assert(map->ndof >= 1 && map->dofs != NULL);

    dof_c = phgDofNew(g, DOF_DEFAULT, 1, "dof_c", DofInterpolation);

    map_c = phgMapCreate(dof_c, NULL);
    _t->Pc = phgMapCreateMat(map, map_c);
    _t->Pi = phgMapCreateMat(map, map);
    _t->Pc->handle_bdry_eqns=FALSE;
    _t->Pi->handle_bdry_eqns=FALSE;
    
    int nd = xi->nd;
    ForAllElementsBegin(g, e) {
      INT I, J, eno = e->index;
      int i, k, N = DofGetNBas(dof_c, e);

      for (k = 0; k < map->ndof; k++) {
	      if (phgXFEMOffside(xi, eno, PartNo(xi, k)))
		  continue;
        for (i = 0; i < N; i++) {
          I = phgXFEMSolverMapE2G(xi, solver, 0, k, e, i);  
          xi->nd = 1;
          J = phgMapE2G(map_c, 0, e, i);
          xi->nd = nd;
          phgMatAddGlobalEntry(_t->Pc, I, J, _F(1.));
          DebugMat
        }
      }

      for (k = 0; k < map->ndof; k++) {
#ifdef DEBUG_P4EST 
        phgPrintf("\t e: %d, \t Mark: %d, \t Macro[%d]: %d\n",
            eno, xi->info[eno].mark, 
            PartNo(xi,k), GetMacro(xi, eno, PartNo(xi, k)));
#endif
      /* remove unmerged offside freedom */
#ifdef PHG_TO_P4EST
        if (xi->info[eno].mark != 0 && GetMacro(xi, eno, k) < 0)
      continue;
#else
        if (xi->info[eno].mark != 0)
      continue;
#endif
        for (i = 0; i < N; i++) {
          I = phgXFEMSolverMapE2G(xi, solver, 0, k, e, i);
          phgMatAddGlobalEntry(_t->Pi, I, I, _F(1.));
          DebugMat
        }
      }

    } ForAllElementsEnd

    phgDofFree(&dof_c);
    phgMapDestroy(&map_c);
    phgMatAssemble(_t->Pc);
    phgMatAssemble(_t->Pi);

    if (_t->dump_mat) {
      phgMatDumpMATLAB(_t->Pc, "Pc", "P0");
      phgMatDumpMATLAB(_t->Pi, "Pi", "P1");
    }

    return;
}


static void
build_ASP_A(SOLVER *solver)
/* builds the stiffness matrices on subspaces V^c and V^i:
 * _t->Ac <--> V^c
 * _t->Ai <--> V^i */
{
  MAT *A = solver->mat_bak != NULL ? solver->mat_bak : solver->mat;
  MAT *temp;

  _t->Ac = phgMapCreateMat(_t->Pc->cmap, _t->Pc->cmap);
  temp = phgMatMat(MAT_OP_T, MAT_OP_N, 1., _t->Pc, A, 0, NULL);
  _t->Ac = phgMatMat(MAT_OP_N, MAT_OP_N, 1., temp, _t->Pc, 0, NULL);
  phgMATProcessEmptyRows(_t->Ac);
  phgMatDestroy(&temp);

  _t->Ai = phgMapCreateMat(_t->Pi->cmap, _t->Pi->cmap);
  temp = phgMatMat(MAT_OP_T, MAT_OP_N, 1., _t->Pi, A, 0, NULL);
  _t->Ai = phgMatMat(MAT_OP_N, MAT_OP_N, 1., temp, _t->Pi, 0, NULL);
  phgMATProcessEmptyRows(_t->Ai);
  phgMatDestroy(&temp);

  if (_t->dump_mat) {
    phgMatDumpMATLAB(_t->Ac, "Ac", "A0");
    phgMatDumpMATLAB(_t->Ai, "Ai", "A1");
  }
  
  return;
}

/*--------------------------------------------------------------------------*/

static int
RegisterOptions(void)
{
    phgOptionsRegisterTitle("\nThe XASP solver options:", "\n", "XASP");

    /* ASP sovler's options */
    phgOptionsRegisterString("-xasp_cycle_type", "XASP cycle type", 
        &cycle_type);
    phgOptionsRegisterNoArg("-xasp_dump_mat", "dump prolongation matrix in .m files",
					&dump_mat);

    /* MG sovler's options */ 
    phgOptionsRegisterInt("-mg_type", "mg type (Non-Full: 1, Full: 2) ",
        &mg_global_params.mg_type);
    phgOptionsRegisterInt("-mg_cycle_type", "cycle type (V: 1 , W: 2)", 
        &mg_global_params.cycle_type);
    phgOptionsRegisterInt("-mg_finest_order", "finest level order of mg", 
        &mg_global_params.finest_order);
    phgOptionsRegisterInt("-mg_coarsest_order", "coarsest level order of mg", 
        &mg_global_params.coarsest_order);
    phgOptionsRegisterInt("-mg_presmooth_iter", "pre-smoothing times of mg", 
        &mg_global_params.presmooth_iter);
    phgOptionsRegisterInt("-mg_postsmooth_iter", "post-smoothing times of mg", 
        &mg_global_params.postsmooth_iter);
    phgOptionsRegisterFloat("-mg_damped", "damping parameters of smoothing", 
        &mg_global_params.damped);
    phgOptionsRegisterString("-mg_approx_solver_opts", "mg's approx solver options",
			  &mg_approx_solver_opts);

    /* auxiliary solver's options */
    phgOptionsRegisterString("-xasp_aux_solver_opts", "auxiliary solver options",
			      &aux_solver_opts);

    return 0;
}

static int
Init(SOLVER *solver)
{
    solver->oem_data = phgCalloc(1, sizeof(OEM_DATA));

    /* copy relavant cmdline arguments */
    _t->cycle_type = strdup(cycle_type);
    _t->dump_mat = dump_mat;
    memcpy(&_mg_param, &mg_global_params, sizeof(MG_PARAM));

    if (aux_solver_opts != NULL)
	_t->aux_solver_opts = strdup(aux_solver_opts);

    if (mg_approx_solver_opts != NULL)
	_t->mg_approx_solver_opts = strdup(mg_approx_solver_opts);

    _t->Ac = NULL;
    _t->Ai = NULL;
    _t->assembled = FALSE;

    return 0;
}

static int
Create(SOLVER *solver)
{
    if (solver->mat->type == PHG_MATRIX_FREE && solver->mat->blocks == NULL) {
	phgError(1, "%s:%d: only ordinary or block matrix is allowed.\n",
						__FILE__, __LINE__);
    }
    return 0;
}

static int
Destroy(SOLVER *solver)
{
    if (solver->oem_data == NULL)
	return 0;

    if (_t->Pc != NULL)
	phgMatDestroy(&_t->Pc);
    if (_t->Pi != NULL)
	phgMatDestroy(&_t->Pi);
    if (_t->Ac != NULL)
	phgMatDestroy(&_t->Ac);
    if (_t->Ai != NULL)
	phgMatDestroy(&_t->Ai);
    if (_t->aux_solver!= NULL)
	phgSolverDestroy(&_t->aux_solver);
    if(_mg_data != NULL)
    {   
  phgMatDestroy(&_mg_data[0].P);
  phgVecDestroy(&_mg_data[0].b);
  phgVecDestroy(&_mg_data[0].x);
  int nl = _mg_data[0].order - _mg_param.coarsest_order + 1;
  for (int i = 1; i < nl; i++) {
      phgMatDestroy(&_mg_data[i].A);
      phgMatDestroy(&_mg_data[i].P);
      phgVecDestroy(&_mg_data[i].b);
      phgVecDestroy(&_mg_data[i].x);
      }
  }
    if (_t->mg_approx_solver != NULL)
  phgSolverDestroy(&_t->mg_approx_solver);

    phgFree(_t->cycle_type);
    phgFree(_t->aux_solver_opts);
    phgFree(_t->mg_approx_solver_opts);

    phgFree(solver->oem_data);
    solver->oem_data = NULL;

    return 0;
}

static int
Assemble(SOLVER *solver)
{
    double t = 0.0, t1;

    if (_t->assembled)
	return 0;

    if (_t->xi == NULL) {
      phgWarning("Shoule call phgSovlerXASPSetXFEMINFO() before \
                    created phgSovlerCreate()!\n");
    }

    _t->assembled = TRUE;

    if (solver->monitor) {
	    t = phgGetTime(NULL);
	    phgPrintf("*** XASP begin setup: \n");
	  }

    /* build transfer matrix for ASP method */
    build_ASP_P(_t->xi, solver);

    /* build subspace stiffness matrix  for ASP method*/
    if (_t->Ac == NULL && _t->Ai == NULL)
        build_ASP_A(solver);

    /* ASP set up done */
	  if (solver->monitor) {
	phgPrintf("    ASP time: %0.4lg \n",(t1 = phgGetTime(NULL)) - t);
	t = t1;
	  }

    /* Creat mg_data */
    INT num_mg_levels = _mg_param.finest_order - _mg_param.coarsest_order + 1;
	  _mg_data = (MG_DATA *)phgCalloc(num_mg_levels, sizeof(MG_DATA));

    /* initialize data[0] with A, b and x */
    _mg_data[0].A = _t->Ac;
    _mg_data[0].b = phgMapCreateVec(_mg_data[0].A->rmap,1);
    _mg_data[0].x = phgMapCreateVec(_mg_data[0].A->rmap,1);
    _mg_data[0].order = _mg_param.finest_order;

    /* local variables level info (fine: mg_level; coarse: mg_level+1) */
    INT mg_level = 0;

    while (mg_level < num_mg_levels-1 ) {
      /* do not change the ordering in this loop */ 
     _mg_data[mg_level+1].order = _mg_data[mg_level].order - 1;
     _mg_data[mg_level+1].P = build_MG_P(_t->xi, solver, mg_level);
	   _mg_data[mg_level+1].A = build_MG_C(solver, mg_level);
	   _mg_data[mg_level+1].b = phgMapCreateVec(_mg_data[mg_level+1].A->rmap,1);
     _mg_data[mg_level+1].x = phgMapCreateVec(_mg_data[mg_level+1].A->rmap,1);
     ++mg_level;
    }

    /* approximate solver of MG */
    phgOptionsPush();
    phgSolverSetDefaultSuboptions();
	  phgOptionsSetOptions("-solver mumps");
	  phgOptionsSetOptions("-mumps_precision double -solver_symmetry sym");
    phgOptionsSetOptions(_t->mg_approx_solver_opts);
    if (_mg_data[mg_level].A != NULL) {
	_t->mg_approx_solver = phgMat2Solver(SOLVER_DEFAULT, _mg_data[mg_level].A);
  _t->mg_approx_solver->monitor = FALSE;
	_t->mg_approx_solver->warn_maxit = FALSE;
 phgVecDestroy(&_t->mg_approx_solver->rhs);
	_t->mg_approx_solver->rhs_updated = TRUE;
    }
    phgOptionsPop();

    /* set up done! */
	  if (solver->monitor) {
	phgPrintf("    R_0 time: %0.4lg \n",(t1 = phgGetTime(NULL)) - t);
	t = t1;
	  }

    /* build auxiliary soler */
    phgOptionsPush();
    phgSolverSetDefaultSuboptions();
	  phgOptionsSetOptions("-solver mumps");
	  phgOptionsSetOptions("-mumps_precision double -solver_symmetry sym");
    phgOptionsSetOptions(_t->aux_solver_opts);
    if (_t->Ai != NULL) {
	_t->aux_solver = phgMat2Solver(SOLVER_DEFAULT, _t->Ai);
  _t->aux_solver->monitor = FALSE;
	_t->aux_solver->warn_maxit = FALSE;
	phgVecDestroy(&_t->aux_solver->rhs);
	_t->aux_solver->rhs_updated = TRUE;
    }
    phgOptionsPop();

    /* set up done! */
	  if (solver->monitor) {
	phgPrintf("    R_1 time: %0.4lg \n",(t1 = phgGetTime(NULL)) - t);
	t = t1;
	  }

    return 0;
}

#if USE_OMP
/* TODO: balance # of non-zeros between threads */
#define ThreadRange(n, tid, start, end)					\
    {									\
	int d = (n) / phgMaxThreads, r = (n) - d * phgMaxThreads;	\
	start = d * (tid) + ((tid) < r ? (tid) : r);			\
 	end = start + d + ((tid) < r ? 1 : 0);				\
    }
#endif	/* USE_OMP */

static VEC *
gauss_seidel(MAT *mat, VEC *rhs, VEC **x_ptr, INT maxit, FLOAT tol,
	     FLOAT *res_ptr, INT verb)
/* scaled symmetric l1-GS relaxation */
{
    VEC *x;
    INT it, i, j, n, *pc = NULL;
    FLOAT *pd = NULL, a, b, res = FLOAT_MAX, *diag1;
#if USE_MPI || USE_OMP
    FLOAT *tmp = NULL;
# if USE_MPI
    INT *pc_offp = NULL;
    FLOAT *pd_offp = NULL, *offp_data = NULL;
# endif	/* USE_MPI */
# if USE_OMP
    FLOAT *xsave, *resp;
# endif	/* USE_OMP */
#endif	/* USE_MPI || USE_OMP */

    x = (x_ptr == NULL ? NULL : *x_ptr);
    if (x == NULL) {
	x = phgMapCreateVec(mat->cmap, rhs->nvec);
	if (x_ptr != NULL)
	    *x_ptr = x;
    }

    if (mat->type == PHG_UNPACKED)
	phgMatPack(mat);

    assert(mat->type == PHG_PACKED);

    if (mat->diag1 == NULL) {
	/* setup off-diagonal L1 norm1 */
	diag1 = mat->diag1 = phgAlloc(mat->rmap->nlocal * sizeof(*mat->diag1));
	if (mat->diag == NULL)
	    phgMatSetupDiagonal(mat);
#if USE_OMP
	if (phgMaxThreads == 1) {
#endif	/* USE_OMP */
#if USE_MPI
	    pd_offp = mat->packed_data + mat->packed_ind[mat->rmap->nlocal];
#endif	/* USE_MPI */
	    for (i = 0; i < mat->rmap->nlocal; i++) {
		a = Fabs(mat->diag[i]);
#if USE_MPI
		/* off-process columns */
		j = mat->rmap->nlocal + i;
		n = (INT)(mat->packed_ind[j + 1] - mat->packed_ind[j]);
		for (j = 0; j < n; j++)
		    a += Fabs(pd_offp[j]);
		pd_offp += n;
#endif	/* USE_MPI */
		assert(a != 0.);
		diag1[i] = 1.0 / a;
	    }
#if USE_OMP
	}
	else {
#if USE_MPI
#pragma omp parallel private(i, j, pd_offp, pc, pd, n, a)
#else	/* USE_MPI */
#pragma omp parallel private(i, j, pc, pd, n, a)
#endif	/* USE_MPI */
{
	    int k;
	    INT startind, endind, l;

	    ThreadRange(x->map->nlocal, phgThreadId, startind, endind)

	    pc = mat->packed_cols + mat->packed_ind[startind];
	    pd = mat->packed_data + mat->packed_ind[startind];
#if USE_MPI
	    j = mat->rmap->nlocal + startind;
	    pd_offp = mat->packed_data + mat->packed_ind[j];
#endif	/* USE_MPI */

# pragma omp for schedule(static)
	    for (k = 0; k < phgMaxThreads; k++) {
		for (i = startind ; i < endind; i++) {
		    a = Fabs(mat->diag[i]);
#if USE_MPI
		    /* off-process columns */
		    j = mat->rmap->nlocal + i;
		    n = (INT)(mat->packed_ind[j + 1] - mat->packed_ind[j]);
		    for (j = 0; j < n; j++)
			a += Fabs(pd_offp[j]);
		    pd_offp += n;
#endif	/* USE_MPI */

		    /* in-process but off-thread columns */
		    n = (INT)(mat->packed_ind[i + 1] - mat->packed_ind[i]);
		    for (j = 0; j < n; j++){
			if ((l = pc[j]) < startind || l >= endind)
			    a += Fabs(pd[j]);
		    }
		    pc += n;
		    pd += n;
    
		    assert(a != 0.);
		    diag1[i] = 1.0 / a;
		} /* i loop */
	    } /* k loop */
} /* omp  parallel*/
	}
#endif	/* USE_OMP */
    } /* mat->diag1 == NULL */
    diag1 = mat->diag1;

#if USE_MPI
    tmp = phgAlloc(x->map->nlocal * sizeof(*tmp));
    if (x->map->nprocs > 1)
	offp_data = phgAlloc(mat->cinfo->rsize * sizeof(*offp_data));
#elif USE_OMP
    tmp = phgAlloc(x->map->nlocal * sizeof(*tmp));
#endif	/* USE_MPI */

#if USE_OMP
    resp = phgAlloc(phgMaxThreads * sizeof(*resp));
    xsave = phgAlloc(x->map->nlocal * sizeof(*xsave));
#endif

#if 0
    double t0 = phgGetTime(NULL);
    maxit = 1000;
#endif

    for (it = 0; it < maxit; it++) {
#if USE_OMP
	if (phgMaxThreads == 1) {
#endif
	/* forward scan */
#if USE_MPI
	    if (x->map->nprocs > 1) {
		phgMapScatterBegin(mat->cinfo, x->nvec, x->data, offp_data);
		phgMapScatterEnd  (mat->cinfo, x->nvec, x->data, offp_data);
	    }
#endif	/* USE_MPI */
	    pc = mat->packed_cols;
	    pd = mat->packed_data;
#if USE_MPI
	    pc_offp = mat->packed_cols + mat->packed_ind[mat->rmap->nlocal];
	    pd_offp = mat->packed_data + mat->packed_ind[mat->rmap->nlocal];
#endif	/* USE_MPI */
	    for (i = 0; i < mat->rmap->nlocal; i++) {
		a = rhs->data[i];
#if USE_MPI
		/* off-process columns */
		j = mat->rmap->nlocal + i;
		n = (INT)(mat->packed_ind[j + 1] - mat->packed_ind[j]);
		for (j = 0; j < n; j++)
		   a -= pd_offp[j] * offp_data[pc_offp[j]];
		tmp[i] = a;
		pc_offp += n;
		pd_offp += n;
#endif	/* USE_MPI */

		/* in-process columns */
		n = (INT)(mat->packed_ind[i + 1] - mat->packed_ind[i]); 
		for (j = 0; j < n; j++)
		    a -= pd[j] * x->data[pc[j]];
		x->data[i] += a * diag1[i];
		pc += n;
		pd += n;
	    }

	    /* backward scan (note: don't exchange and use the new off_p data
	     * here, or the convergence will be slower) */
	    res = 0.0;
	    for (i = mat->rmap->nlocal - 1; i >= 0; i--) {
#if USE_MPI
		a = tmp[i];
#else	/* USE_MPI */
		a = rhs->data[i];
#endif	/* USE_MPI */

		/* in-process columns */
		n = (INT)(mat->packed_ind[i + 1] - mat->packed_ind[i]); 
		pc -= n;
		pd -= n;
		for (j = n - 1; j >= 0; j--)
		    a -= pd[j] * x->data[pc[j]];
		b = x->data[i];
		x->data[i] += a * diag1[i];
		b = Fabs(x->data[i] - b);
		if (res <= b)
		    res = b;
	   }
#if USE_OMP
	}
	else {
	    /* forward scan */
#if USE_MPI
	    if (x->map->nprocs > 1) {
		phgMapScatterBegin(mat->cinfo, x->nvec, x->data, offp_data);
		phgMapScatterEnd  (mat->cinfo, x->nvec, x->data, offp_data);
	    }
#endif	/* USE_MPI */

	    if (x->map->nlocal > 0)
	 	memcpy(xsave, x->data, x->map->nlocal * sizeof(*xsave));

#if USE_MPI
# pragma omp parallel default(shared) \
	private(a, b, i, j, pc_offp, pd_offp, n, pc, pd)
#else	/* USE_MPI */
# pragma omp parallel default(shared) private(a, b, i, j, n, pc, pd)
#endif	/* USE_MPI */
{ 
	    int k;
	    INT startind, endind, l;

	    ThreadRange(x->map->nlocal, phgThreadId, startind, endind)

	    pc = mat->packed_cols + mat->packed_ind[startind];
	    pd = mat->packed_data + mat->packed_ind[startind];
#if USE_MPI
	    j = mat->rmap->nlocal + startind;
	    pc_offp = mat->packed_cols + mat->packed_ind[j];
	    pd_offp = mat->packed_data + mat->packed_ind[j];
#endif	/* USE_MPI */

#pragma omp for schedule(static)
	    for (k = 0; k < phgMaxThreads; k++) {
		for (i = startind; i < endind; i++) {
		    a = rhs->data[i];
#if USE_MPI
		    /* off-process columns */
		    j = mat->rmap->nlocal + i;
		    n = (INT)(mat->packed_ind[j + 1] - mat->packed_ind[j]);
		    for (j = 0; j < n; j++)
			a -= pd_offp[j] * offp_data[pc_offp[j]];
		    pc_offp += n;
		    pd_offp += n;
#endif	/* USE_MPI */

		    /* in-process columns */
		    n = (INT)(mat->packed_ind[i + 1] - mat->packed_ind[i]);
		    b = 0.0;
		    for (j = 0; j < n; j++) {
			if ((l = pc[j]) < startind || l >= endind) {
			    /* off-thread column */
			    a -= pd[j] * xsave[l];
			}
			else {
			    /* in-thread column */
			    b -= pd[j] * x->data[l];
			}
		    }
		    tmp[i] = a;
		    x->data[i] += (a + b) * diag1[i];
		    pc += n;
		    pd += n;
		}
	    } /* k loop */

	    resp[phgThreadId] = 0.0;

#pragma omp for schedule(static)
	    for (k = 0; k < phgMaxThreads; k++) {
		for (i = endind - 1; i >= startind; i--) {
		    a = tmp[i];
		    /* in-process columns */
		    n = (INT)(mat->packed_ind[i + 1] - mat->packed_ind[i]);
		    pc -= n;
		    pd -= n;
		    for (j = n - 1; j >= 0; j--) {
			if ((l = pc[j]) >= startind && l < endind)
			    a -= pd[j] * x->data[l];
		    }
		    b = x->data[i];
		    x->data[i] += a * diag1[i];
		    b = Fabs(x->data[i] - b);
		    if (resp[phgThreadId] <= b)
			resp[phgThreadId] = b;
		}
	    }
} /* omp parallel */
	    res = resp[0];
	    for (i = 1; i < phgMaxThreads; i++)
		if (res < resp[i]) 
		    res = resp[i]; 
	} /* if phgMaxThreads == 1 */
#endif
	if (verb)
           phgPrintf("\n***  Nits: %d ,  | b - Ax |_linf: %6le\n\n", 
			   it, (double)res);

	if (res <= tol)
	    break;
    } /* it - loop */

#if 0
    printf("time = %lg\n", phgGetTime(NULL) - t0);
    MPI_Finalize();
    exit(0);
#endif

#if USE_MPI
    phgFree(tmp);
    phgFree(offp_data);
#elif USE_OMP
    phgFree(tmp);
#endif

#if USE_OMP
    phgFree(xsave);
    phgFree(resp);
#endif	/* USE_OMP */

    if (res_ptr != NULL)
	*res_ptr = res;

    return x;
}

static void
mult_prec(SOLVER *solver, VEC *r, VEC *x, const char *cycle)
{
    VEC *r0 = NULL, *r1 = NULL, *x1 = NULL;
    FLOAT l2_norm;

    while (*cycle != '\0') {
	if (*cycle < '0' || *cycle > '2')
	    phgError(1, "invalid XASP cycle type string: %s\n", _t->cycle_type);
	switch (*(cycle++) - '0') {
	    case 0:	/* smoothing */
		if (solver->monitor) {
		phgVecCopy(r, &r0);
		phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);
	        l2_norm = phgVecNorm2(r0, 0, NULL);
	        phgPrintf("\n*** Before  l1-GS  | r |_l2: %6le\n\n", (double)l2_norm);
		}
		gauss_seidel(solver->mat, r, &x, _mg_param.finest_order, 0., NULL, 0);
		if (solver->monitor) {
			phgVecCopy(r, &r0);
			phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);
			l2_norm = phgVecNorm2(r0, 0, NULL);
			phgPrintf("\n*** After   l1-GS  | r |_l2: %6le\n\n", (double)l2_norm);
		}
		break;
	    case 1:	/* space V^c correction by 1 step MG */
		phgVecCopy(r, &r0);
		phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);

    /* intial correction vector */ 
		if (x1 == NULL || x1->map != _t->Pc->cmap) {
		    phgVecDestroy(&x1);
		    x1 = phgMapCreateVec(_t->Pc->cmap, 1);
		} else 
        bzero(x1->data, x1->map->nlocal * sizeof(*x1->data));

     /* initial rhs vector */ 
		if (r1 != NULL && r1->map != _t->Pc->cmap)
		    phgVecDestroy(&r1);

      if (solver->monitor) {
	      l2_norm = phgVecNorm2(r0, 0, NULL);
	      phgPrintf("\n*** Before  PMG    | r |_l2: %6le \n\n", (double)l2_norm);
      }

	      phgMatVec(MAT_OP_T, 1.0, _t->Pc, r0, 0.0, &r1);

      if (solver->monitor) {
	      l2_norm = phgVecNorm2(r1, 0, NULL);
              phgPrintf("\n***            | Pc^Tr |_l2: %6le\n\n", (double)l2_norm);
      }

    /* give rhs of finest level */ 
    phgVecCopy(r1, &_mg_data[0].b);

    /* one mg cycle */ 
    switch(_mg_param.mg_type) {
      case 1:
        nrmg(solver);
        break;
      case 2:
        fmg(solver);
        break;
      default:
	      phgError(1, "%s:%d: invalid mg_type (%d).\n",
		            __FILE__, __LINE__, _mg_param.mg_type);
    }

    /* Save solution vector and return */
    phgVecCopy(_mg_data[0].x, &x1);

		phgMatVec(MAT_OP_N, 1.0, _t->Pc, x1, 1.0, &x);
    if (solver->monitor) {
	    phgVecCopy(r, &r0);
	    phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);
	    l2_norm = phgVecNorm2(r0, 0, NULL);
	    phgPrintf("\n*** After   PMG    | r |_l2: %6le\n\n", (double)l2_norm);
    }
		break;
	    case 2:	/* space V^i correction by aux_solver */
		if (_t->aux_solver== NULL)
		    break;
		phgVecCopy(r, &r0);
		phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);

    /* initial correction vector */
		if (x1 == NULL || x1->map != _t->Pi->cmap) {
		    phgVecDestroy(&x1);
		    x1 = phgMapCreateVec(_t->Pi->cmap, 1);
		} else 
		    bzero(x1->data, x1->map->nlocal * sizeof(*x1->data));

    /* initial rhs vector */
		if (r1 != NULL && r1->map != _t->Pi->cmap) 
		    phgVecDestroy(&r1);

      if (solver->monitor) {
	      l2_norm = phgVecNorm2(r0, 0, NULL);
	      phgPrintf("\n*** Before  mumps  | r |_l2: %6le\n\n", (double)l2_norm);
      }

		phgMatVec(MAT_OP_T, 1.0, _t->Pi, r0, 0.0, &r1);

      if (solver->monitor) {
	      l2_norm = phgVecNorm2(r1, 0, NULL);
              phgPrintf("\n***            | Pi^Tr |_l2: %6le\n\n", (double)l2_norm);
      }
		_t->aux_solver->rhs = r1;
		_t->aux_solver->rhs->assembled = TRUE;
		phgSolverVecSolve(_t->aux_solver, FALSE, x1);
		_t->aux_solver->rhs = NULL;
		phgMatVec(MAT_OP_N, 1.0, _t->Pi, x1, 1.0, &x);

    if (solver->monitor) {
	    phgVecCopy(r, &r0);
	    phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &r0);
	    l2_norm = phgVecNorm2(r0, 0, NULL);
	    phgPrintf("\n*** After   mumps  | r |_l2: %6le\n\n", (double)l2_norm);
    }
    break;
	}	/* case */
    }	/* while */

    if (r0 != NULL)
	phgVecDestroy(&r0);
    if (r1 != NULL)
	phgVecDestroy(&r1);
    if (x1 != NULL)
	phgVecDestroy(&x1);
}

static int
Solve(SOLVER *solver, VEC *x, BOOLEAN destroy)
{
    char *p = _t->cycle_type, *q;
    VEC *y0 = NULL, *x0 = NULL;
    FLOAT res = FLOAT_MAX, ib_norm = 0.0, ires0 = 0.0, tol = 0.0;

    Assemble(solver);

    if (solver->btol > 0. || solver->rtol > 0. ||
	solver->atol > 0. || solver->monitor) {
	ib_norm = phgVecNorm2(solver->rhs, 0, NULL);
	if (ib_norm == 0.0) {
	    solver->nits = 0;
	    solver->residual = 0.;
	    bzero(x->data, x->nvec * x->map->nlocal * sizeof(*x->data));
	    return 0;
	}
	tol = solver->btol * ib_norm;
	if (tol < solver->atol)
	    tol = solver->atol;
	ib_norm = 1.0 / ib_norm;	/* inversed rhs norm */
    }

    solver->nits = 0;
    while (solver->nits < solver->maxit) {
	solver->nits++;

	if ((q = strchr(p, '+')) == NULL) {
	    mult_prec(solver, solver->rhs, x, p);
	}
	else {
	    phgVecCopy(solver->rhs, &y0);
	    phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &y0);
	    x0 = phgMapCreateVec(y0->map, y0->nvec);
	    while (*p != '\0') {
		if (q != NULL)
		    *q = '\0';
		bzero(x0->data, sizeof(*x0->data) * x0->map->nlocal);
		mult_prec(solver, y0, x0, p);
		phgVecAXPBY(1.0, x0, 1.0, &x);
		if (q == NULL)
		    break;
		*q = '+';
		q = strchr(p = q + 1, '+');
	    }
	}

	if (solver->rtol > 0.0 || tol > 0.0 || solver->monitor) {
	    phgVecCopy(solver->rhs, &x0);
	    phgMatVec(MAT_OP_N, -1.0, solver->mat, x, 1.0, &x0);
	    res = phgVecNorm2(x0, 0, NULL);
	    if (ires0 == 0.0) {
		ires0 = solver->rtol * res;
		if (tol < ires0)
		    tol = ires0;
		ires0 = (res == 0. ? 1.0 : 1.0 / res);	/* inversed res norm */
	    }

	    if (solver->monitor)
		phgPrintf("*** XASP % 5d   %12le   %12le   %12le\n", solver->nits,
        (double)res, (double)(res * ires0), (double)(res * ib_norm));
	    if (res <= tol)
		break;
	}
    }

    phgVecDestroy(&x0);
    phgVecDestroy(&y0);

    if (destroy) {
	Destroy(solver);
	phgMatDestroy(&solver->mat);
    }

    solver->residual = res/*_b*/;

    return solver->nits;
}

OEM_SOLVER phgSolverXASP_ = {
    "xasp", RegisterOptions,
    Initialize, Finalize, Init, Create, Destroy, AddMatrixEntries,
    AddRHSEntries, Assemble, SetPC, Solve, NULL, NULL, NULL,
    M_UNSYM, TRUE, TRUE, TRUE, FALSE
};

#define SOLVER_XASP (&phgSolverXASP_)
