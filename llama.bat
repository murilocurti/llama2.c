cls
x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -shared -o llama.dll llama.c transformer.c win.c -lm
x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run2.exe -I. run2.c win.c -L. -llib
run2 C:\Users\MuriloCurti\source\repos\.murilocurti\stories15M.bin