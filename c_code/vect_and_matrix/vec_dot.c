int main() {
    int a[4];
    int b[4];
    int s;
    s = 0;
    for (int i = 0; i < 4; i++) {
        s = s + a[i] * b[i];
    }
    return s;
}