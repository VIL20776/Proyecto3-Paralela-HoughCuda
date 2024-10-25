all: hough.bin pgm.o

hough.bin:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough.bin

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o

clean:
	rm *.bin *.o