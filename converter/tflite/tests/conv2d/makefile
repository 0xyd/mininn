GCC := gcc

all: cpfiles build

cpfiles:
	rm -f ikann.* 
	cp -a ../../../../ikann.* .
	rm -f utils.*
	cp -a ../../../../utils.* .

build:
	gcc -o hello ikann.c utils.c main.c model.c -lm