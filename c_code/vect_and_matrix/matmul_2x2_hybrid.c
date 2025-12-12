int main() {
    int A[2][2] = {{1, 2}, {3, 4}};
    int B[2][2] = {{5, 6}, {7, 8}};
    int C[2][2];

    int x = 5;
    int y = 7;
    int z = x + y;

    for (int i = 0; i < 2; i = i + 1) {
        for (int j = 0; j < 2; j = j + 1) {
            C[i][j] = 0;
            for (int k = 0; k < 2; k = k + 1) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }

    return z;
}