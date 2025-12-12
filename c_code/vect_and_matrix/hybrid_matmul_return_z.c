int main() {
    int x = 5;
    int y = 7;
    int z = x + y;

    int A[2][2] = {{1, 2}, {0, 1}};
    int B[2][2] = {{1, 0}, {3, 1}};
    int C[2][2];

    for (int i = 0; i < 2; i = i + 1) {
        for (int j = 0; j < 2; j = j + 1) {
            C[i][j] = 0;
            for (int k = 0; k < 2; k = k + 1) {
                C[i][j] = C[i][j] + (A[i][k] * B[k][j]);
            }
        }
    }

    // Ibrido: somma scalare + elemento della matrice risultante
    return z + C[1][1];
}