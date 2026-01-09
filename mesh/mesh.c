#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

int main(int argc, char *argv[])
{
  /* input values */
  int sec_num[3]; 
  double box[3][2];
  double *input_coords[3]; 
  int count = 1;    
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < 2; i++) {
      box[d][i] = atof(argv[count]);
      count++;
    }
    sec_num[d] = atoi(argv[count]);
    count++;
    input_coords[d] = calloc(sec_num[d]+2, sizeof(double));
    input_coords[d][0] = box[d][0];
    input_coords[d][sec_num[d]+1] = box[d][1];
    for (int i = 1; i <= sec_num[d]; i++) {
      input_coords[d][i] = atof(argv[count]);
      count++;
    }
  }
 
  /* marks, references and coordinates w.r.t element */
  int N_ELEMENTS = (sec_num[0]+1)*(sec_num[1]+1)*(sec_num[2]+1);
  int N_VERTICES = (sec_num[0]+2)*(sec_num[2]+2)*(sec_num[2]+2);
  int n_elem, n_vert = 1; 
  int *mark_elem[8];
  double  *coords[3];
  for (int i = 0; i < 8; i++)
    mark_elem[i]  = calloc(N_ELEMENTS, sizeof(int));
  for (int d = 0; d < 3; d++)
    coords[d] = calloc(N_VERTICES, sizeof(double));
  int *ref_elem = (int*)malloc(N_ELEMENTS * sizeof(int));   
  for (int i = 0; i <=  sec_num[2]; i++) 
  {
    for (int j = 0; j <= sec_num[1]; j++)
    {
      for (int k = 0; k <= sec_num[0]; k++)
      {
        /* element index */
        n_elem = i*(sec_num[2]+1)*(sec_num[1]+1)+j*(sec_num[1]+1)+k;
        /* element vertex mark */
        mark_elem[0][n_elem] = n_vert;
        mark_elem[1][n_elem] = mark_elem[0][n_elem]+1;
        mark_elem[2][n_elem] = mark_elem[1][n_elem]+sec_num[0]+1;
        mark_elem[3][n_elem] = mark_elem[2][n_elem]+1;
        mark_elem[4][n_elem] = mark_elem[0][n_elem]+(sec_num[0]+2)*(sec_num[1]+2);
        mark_elem[5][n_elem] = mark_elem[4][n_elem]+1;
        mark_elem[6][n_elem] = mark_elem[5][n_elem]+sec_num[0]+1;
        mark_elem[7][n_elem] = mark_elem[6][n_elem]+1;
        /* element references */
        ref_elem[n_elem] = (i + j + k ) % 2;
        n_vert++;
      }
      n_vert++;
    }
    n_vert += sec_num[0]+2;
  }

  n_vert = 0;
  for (int i = 0; i < sec_num[2]+2; i++)
  {
    for (int j = 0; j < sec_num[1]+2; j++)
    {
      for (int k = 0; k < sec_num[0]+2; k++)
      {
        coords[0][n_vert] = input_coords[0][k];
        coords[1][n_vert] = input_coords[1][j];
        coords[2][n_vert] = input_coords[2][i];
        n_vert++;
      }
    }
  }

  /* marks w.r.t cross face */
  int N_FACES = 0;
  int *mark_face[3];
  for (int i = 0; i < 3; i++)
    mark_face[i]  = calloc(N_ELEMENTS, sizeof(int));
  for (n_elem = 0; n_elem < N_ELEMENTS; n_elem++)
  {
    for (int i = 0; i < 3; i++)
    {
      int local_index;
      local_index = (i  == 2) ? i+1 : i+2;
      int vert_index = mark_elem[local_index][n_elem];
      double vert_coord = coords[i][vert_index-1];
      if (vert_coord != box[i][0] && vert_coord != box[i][1]) 
      {
        N_FACES++;
        mark_face[i][n_elem] = mark_elem[local_index][n_elem];  
      }
    }
  }

  /* print information */ 
  printf("MeshVersionFormatted 1\n");
  printf("Dimension 3         \n");

  printf("\nVertices %d\n", N_VERTICES);
  for(n_vert = 0; n_vert  < N_VERTICES; n_vert ++)
    printf("%12.9f %12.9f %12.9f 0\n", coords[0][n_vert],
           coords[1][n_vert], coords[2][n_vert]);

  printf("\nHexahedra %d \n", N_ELEMENTS);
  for(n_elem = 0; n_elem < N_ELEMENTS; n_elem++){
    printf("%d %d %d %d %d %d %d %d %d\n",mark_elem[0][n_elem],
           mark_elem[1][n_elem],mark_elem[2][n_elem],
           mark_elem[3][n_elem],mark_elem[4][n_elem],
           mark_elem[5][n_elem],mark_elem[6][n_elem],
           mark_elem[7][n_elem],ref_elem[n_elem]);
  }

  /* cross face */ 
  printf("\n# quadrilaterals are used to specify boundary types (default: 0)\n");
  printf("Quadrilaterals %d\n", N_FACES);
  for(n_elem = 0; n_elem < N_ELEMENTS; n_elem++){
    for (int i = 0; i < 3; i++)
    {
      if (mark_face[i][n_elem] != 0) 
      {
        int local_index;
        local_index = (i  == 2) ? i + 1 : i + 2;
        int l, m;
        l = (i == 0) ? local_index + 2 : local_index + 1;
        m = (i == 2) ? l + 1 : l + 2;
        printf("%d %d %d %d %d\n",mark_face[i][n_elem], 
            mark_elem[l][n_elem], 
            mark_elem[m][n_elem], 
            mark_elem[7][n_elem], 1); 
      }
    }
  }

  return 0;
}
