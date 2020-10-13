GCC := gcc

all: cpfiles build

cpfiles:
	rm -f ikann.* 
	cp -a ../ikann.* .

build:
	gcc -o mlp ikann.c mlp.c -lm