// c_code/test_confronto/t_vec_add.c
int main() {
  int a[4] = {1, 2, 3, 4};
  int b[4] = {4, 3, 2, 1};
  int c[4];

  for (int i = 0; i < 4; i++) {
    c[i] = a[i] + b[i];
  }

  return c[2];
}