# pragma once

class PGMImage {
public:
    unsigned char* pixels;
    int x_dim;
    int y_dim;

    PGMImage(const char* filename);
    ~PGMImage();
};