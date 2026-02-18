int main() {
    int A[2][2] = {{1, 0}, {0, 2}};
    int B[2][2] = {{3, 0}, {0, 4}};
    int C[2][2] = {{0, 0}, {0, 0}};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int s = 0;
            for (int k = 0; k < 2; k++) {
                s = s + A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }

    return C[0][0];
}
