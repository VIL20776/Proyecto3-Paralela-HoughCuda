#include "pgm.h"

#include <iostream>
#include <fstream>
#include <vector>


PGMImage::PGMImage(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error al abrir el archivo " << filename << std::endl;
        exit(1);
    }

    std::string line;
    file >> line; // Leer el encabezado "P5"
    if (line != "P5") {
        std::cerr << "Formato de archivo no soportado" << std::endl;
        exit(1);
    }

    file >> x_dim >> y_dim;
    int max_val;
    file >> max_val;
    file.ignore(); // Ignorar el carácter de nueva línea

    pixels = new unsigned char[x_dim * y_dim];
    file.read(reinterpret_cast<char*>(pixels), x_dim * y_dim);

    file.close();
}

PGMImage::~PGMImage() {
    delete[] pixels;
}
