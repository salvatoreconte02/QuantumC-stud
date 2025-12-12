int main() {
    int x = 5;
    int y = 7;
    int z = x + y;          // parte scalare -> 12

    int a[2]= {1, 2};
    int b[2]= {3, 4};
    int c[2];

    for (int i = 0; i < 2; i = i + 1) {
        c[i] = a[i] + b[i]; // parte vettoriale -> c = {4, 6}
    }

    return z;        // 12 
}