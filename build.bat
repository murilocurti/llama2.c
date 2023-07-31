cls
make win64lib
x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o sample sample.c -L. -llib
sample