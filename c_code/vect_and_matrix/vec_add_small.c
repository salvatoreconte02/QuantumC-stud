#define N 2

int main() {
    int a[N];
    int b[N];
    int c[N];

    for (int i = 0; i < N; i = i + 1) {
        c[i] = a[i] + b[i];
    }

    return 0;
}