int main() {
    int a[2] = {1, 2};
    int b[2] = {3, 4};

    int acc = 0;

    for (int i = 0; i < 2; i = i + 1) {
        acc = acc + a[i] * b[i];
    }

    return acc;
}