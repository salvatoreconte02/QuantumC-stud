int main() {
    int a = 3;
    int b = 6;
    int c = a - b;   // atteso: -3 (in 2's complement)
    int d = c + 1;   // atteso: -2
    return d;
}