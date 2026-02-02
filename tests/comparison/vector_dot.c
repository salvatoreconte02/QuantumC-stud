int main() {
  int a[4] = {1, 2, 3, 4};
  int b[4] = {4, 3, 2, 1};
  int acc = 0;

  for (int i = 0; i < 4; i++) {
    acc = acc + a[i] * b[i];
  }

  return acc;
}