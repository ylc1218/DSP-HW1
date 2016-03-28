.PHONY: all clean

CFLAGS+=
LDFLAGS+=-lm     # link to math library

TARGET=train test

all: $(TARGET)

train: train.cpp
	g++ -O2 train.cpp -o train

test: test.cpp
	g++ -O2 test.cpp -o test

clean:
	rm $(TARGET)

