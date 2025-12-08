#define N 2

int main() {
    int a[N];
    int b[N];
    int c[N];

    int s;
    int x = 5;
    int y = 7;
    int z = x + y;      // parte scalare

    s = 0;
    for (int i = 0; i < N; i = i + 1) {
        s = s + a[i] * b[i];    // VecDot
    }

    for (int i = 0; i < N; i = i + 1) {
        c[i] = a[i] + b[i];     // VecAdd
    }

    return z + s;
}