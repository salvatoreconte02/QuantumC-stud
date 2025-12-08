int main() {
    int a[4][4];
    int b[4][4];
    int c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            c[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }

    return 0;
}