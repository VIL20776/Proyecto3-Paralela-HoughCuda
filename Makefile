all: hough.bin houghConst.bin houghShared.bin pgm.o bmp.o

houghConst.bin:	houghConst.cu pgm.o bmp.o
	nvcc houghConst.cu pgm.o bmp.o -o houghConst.bin -g

houghShared.bin: houghShared.cu pgm.o bmp.o
	nvcc houghShared.cu pgm.o bmp.o -o houghShared.bin -g

hough.bin:	houghBase.cu pgm.o bmp.o
	nvcc houghBase.cu pgm.o bmp.o -o hough.bin -g

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o -g

bmp.o: common/bmp.cpp
	g++ -c common/bmp.cpp -o bmp.o -g

clean:
	rm *.bin *.o