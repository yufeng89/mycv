CC = gcc
CXX = g++

CFLAG = -std=c++11
CFLAG += `pkg-config --cflags opencv`
CFLAG += `pkg-config --cflags gtk+-2.0 gthread-2.0`

LFLAG = `pkg-config --libs opencv`
LFLAG += `pkg-config --libs gtk+-2.0 gthread-2.0`
LFLAG += -lpthread

aim = demo
test = image

$(aim): main.o subPixelCal.o windowParam.o
	$(CXX) main.o subPixelCal.o windowParam.o -o $(aim) $(LFLAG)

main.o: main.cpp
	$(CXX) -c main.cpp $(CFLAG)

subPixelCal.o: subPixelCal.cpp subPixelCal.h zernikeMask.h
	$(CXX) -c subPixelCal.cpp $(CFLAG)

windowParam.o: windowParam.cpp windowParam.h
	$(CXX) -c windowParam.cpp $(CFLAG)

$(test): image.cpp
	$(CXX) image.cpp -o $(test) $(CFLAG) $(LFLAG)

all:
	make $(aim)

run:
	make $(aim)
	./$(aim)

clean:
	rm -rf *.o demo image
