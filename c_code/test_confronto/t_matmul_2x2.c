// c_code/test_confronto/t_matmul_2x2.c
int main() {
  int A[2][2] = {{1, 2}, {3, 4}};
  int B[2][2] = {{2, 0}, {1, 2}};
  int C[2][2];

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      int s = 0;
      for (int k = 0; k < 2; k++) {
        s = s + A[i][k] * B[k][j];
      }
      C[i][j] = s;
    }
  }

  return C[1][0];
}