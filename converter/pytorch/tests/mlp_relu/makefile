GCC := gcc

all: cpfiles build

cpfiles:
	rm -f ikann.* 
	cp -a ../../../../ikann.* .

build:
	gcc -o hello ikann.c main.c model.c -lm