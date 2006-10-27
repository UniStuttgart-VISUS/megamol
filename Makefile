# Applications:
MAKE = make
SHELL = /bin/bash

# Project directories to make:
ProjectDirs = base sys math graphics gl net test glutTest

################################################################################

all:
	@for dir in $(ProjectDirs); do		\
		pushd $$dir;					\
		$(MAKE)	$@ || exit 1;			\
		popd;							\
	done

sweep:
	@for dir in $(ProjectDirs); do		\
		pushd $$dir;					\
		$(MAKE)	$@ || exit 1;			\
		popd;							\
	done
	
clean:	
	@for dir in $(ProjectDirs); do		\
		pushd $$dir;					\
		$(MAKE)	$@ || exit 1;			\
		popd;							\
	done
	
rebuild: clean all

.PHONY: all clean sweep rebuild