int main() {
    int a[4] = {1, 0, 0, 2};
    int b[4] = {3, 0, 4, 0};
    int s = 0;

    for (int i = 0; i < 4; i++) {
        s = s + a[i] * b[i];
    }

    return s;
}
