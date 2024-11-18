#pragma once

#include <cstdint>

struct Color
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

class BMPImage {
private:

#pragma pack(push,1)
struct Header {
    unsigned char signature [2];
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
} header;
struct Info {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    uint32_t xPixelsPerM;
    uint32_t yPixelsPerM;
    uint32_t colorsUsed;
    uint32_t importantColors;
} info;
#pragma pack(pop)

Color *pixels;

public:
BMPImage(Color* data, unsigned int width, unsigned int height);
~BMPImage();

void save(const char* name);

};
