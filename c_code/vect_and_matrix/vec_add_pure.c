int main() {
    int a[2] = {1, 2};
    int b[2] = {3, 4};
    int c[2];

    for (int i = 0; i < 2; i = i + 1) {
        c[i] = a[i] + b[i] ;
    }

    return c[1];
}