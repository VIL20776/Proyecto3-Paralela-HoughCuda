all: hough.bin pgm.o bmp.o

hough.bin:	houghBase.cu pgm.o bmp.o
	nvcc houghBase.cu pgm.o bmp.o -o hough.bin -g

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o -g

bmp.o: common/bmp.cpp
	g++ -c common/bmp.cpp -o bmp.o -g

clean:
	rm *.bin *.o