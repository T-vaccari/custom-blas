UNAME_S := $(shell uname -s)


CFLAGS = -O3 -Wall

ifeq ($(UNAME_S),Darwin)
    CC = clang
    CFLAGS += -DACCELERATE_NEW_LAPACK -mcpu=apple-m2 -Rpass=loop-vectorize
    LDFLAGS = -framework Accelerate
else
    CC = gcc
    LDFLAGS = -lopenblas -lm
endif

custom_blas: custom_blas.c
	$(CC) $(CFLAGS) custom_blas.c -o custom_blas $(LDFLAGS)

clean:
	rm -f custom_blas
