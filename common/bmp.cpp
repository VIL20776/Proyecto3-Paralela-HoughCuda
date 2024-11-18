#include "bmp.h"

#include <fstream>
#include <cstring>

BMPImage::BMPImage(Color *data, unsigned int width, unsigned int height) {
    // Header
    header = {
        .signature = {'B', 'M'},
        .fileSize = 54 + (width * height * 3),
        .reserved = 0,
        .dataOffset = 54,
    };
    // Info header
    info = {
        .size = 40,
        .width = width,
        .height = height,
        .planes = 1,
        .bitsPerPixel = 24,
        .compression = 0,
        .imageSize = width * height * 3,
        .xPixelsPerM = 0,
        .yPixelsPerM = 0,
        .colorsUsed = 0,
        .importantColors = 0
    };

    pixels = new Color[width * height];
    std::memcpy(pixels, data, sizeof(Color) * width * height);
}

BMPImage::~BMPImage() {
    delete[] pixels;
}

void BMPImage::save(const char* name) {
    std::ofstream of(name, std::ios_base::binary);
    of.write((const char*) &header, sizeof(header));
    of.write((const char*) &info, sizeof(info));
    of.write((const char*) pixels, info.imageSize);
    of.close();
}