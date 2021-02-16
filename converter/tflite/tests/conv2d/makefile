GCC := gcc

ifeq ($(BASEDIR),)
BASEDIR := .
endif

ifeq ($(IKANN_PATH),)
IKANN_PATH := ../../../..
endif

ifeq ($(MAIN_FILE),)
MAIN_FILE := main.c
endif

all: cpfiles build

cpfiles:
	rm -f $(BASEDIR)/ikann.* 
	rm -f $(BASEDIR)/utils.*
	cp -a $(IKANN_PATH)/utils.* $(BASEDIR)
	cp -a $(IKANN_PATH)/ikann.* $(BASEDIR)

build:
	gcc -o $(BASEDIR)/hello $(BASEDIR)/ikann.c $(BASEDIR)/utils.c $(BASEDIR)/$(MAIN_FILE) $(BASEDIR)/model.c -lm 