int main() {
    int a[2] = {0, 1};
    int b[2] = {1, 0};
    int c[2];
    for (int i = 0; i < 2; i++) {
        c[i] = a[i] + b[i];
    }
    return 0;
}
