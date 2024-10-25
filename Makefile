all: hough pgm.o

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o

clean:
	rm -f hough pgm.o