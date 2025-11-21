int main() {
    int a[4];
    int b[4];
    int c[4];
    int s;

    // Prodotto scalare: s = Σ a[i] * b[i]
    s = 0;
    for (int i = 0; i < 4; i++) {
        s = s + a[i] * b[i];
    }

    // Somma di vettori: c[i] = a[i] + b[i]
    for (int i = 0; i < 4; i++) {
        c[i] = a[i] + b[i];
    }

    return s;
}