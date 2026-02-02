#define N 2

int main() {
    int a[N] = {1, 2};
    int b[N] = {3, 4};

    int x = 5;
    int y = 7;
    int z = x + y;

    int s = 0;
    for (int i = 0; i < N; i = i + 1) {
        s = s + a[i] * b[i];
    }

    return s;
}
