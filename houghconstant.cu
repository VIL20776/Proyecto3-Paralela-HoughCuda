/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
// Librerías necesarias para el desarrollo del proyecto.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "common/pgm.h"

// Constantes globales importantes para la transformada de Hough.
const int degreeInc = 2;
const int degreeBins = (180 / degreeInc);
const int rBins = 100;
const float radInc = ((degreeInc * M_PI) / 180);

// Función CPU_HoughTran, que calcula la transformada de Hough de forma secuencial.
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {

  // Cálculo del r máximo a utilizar y asignación de memoria.
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, (sizeof(int) * rBins * degreeBins));

  // Cálculo del centro de la imagen para utilizar como origen.
  int xCent = (w / 2);
  int yCent = (h / 2);
  float rScale = ((2 * rMax) / rBins);

  // Iteración sobre los pixeles de la imagen.
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      // Cálculo del índice a utilizar en la iteración.
      int idx = ((j * w) + i);

      // Verificación de que el pixel de la imagen no sea negro para calcular la transformada.
      if (pic[idx] > 0) {

        // Coordenada en x y en y.
        int xCoord = (i - xCent);
        int yCoord = (yCent - j);

        // Theta inicial a utilizar como prueba.
        float theta = 0;

        // Iteración sobre el rango de ángulos a utilizar.
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

          // Cálculo del valor de r de la iteración.
          float r = ((xCoord * cos(theta)) + (yCoord * sin(theta)));

          // Cálculo del índice a probar.
          int rIdx = ((r + rMax) / rScale);

          // Aumento del índice que se utilizó.
          (*acc)[rIdx * degreeBins + tIdx]++;

          // Aumento de theta según el incremento configurado.
          theta += radInc;
        }
      }
    }
  }
}

// Declaración de variables en memoria constante.
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Kernel del programa utilizado para paralelizar el proceso de cálculo de la transformada.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {

  // Cálculo y verificación de que el ID sea válido.
  int gloID = threadIdx.x + blockIdx.x * blockDim.x;
  if (gloID > (w * h)) return;

  // Cálculo del centro de la imagen.
  int xCenter = (w / 2);
  int yCenter = (h / 2);

  // Coordenada o pixel a utilizar en el presente kernel.
  int xCoord = ((gloID % w) - xCenter);
  int yCoord = (yCenter - (gloID / w));

  // Verificación de que el pixel iterado no sea negro.
  if (pic[gloID] > 0) {

    // Ciclo for que itera en los ángulos a probar por el kernel.
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

      // Cálculo del r a probar y adición del resultado.
      float r = ((xCoord * d_Cos[tIdx]) + (yCoord * d_Sin[tIdx]));
      int rIdx = ((r + rMax) / rScale);

      // Barrera para sincronizar los hilos dentro del bloque.
      __syncthreads();

      // Actualización del acumulador.
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

// Función para dibujar las líneas detectadas en la imagen original y guardarla
void drawAndSaveLines(const char *outputFileName, unsigned char *originalImage, int w, int h, int *h_hough, float rScale, float rMax, int threshold) {

  // Instancia de la imagen a crear.
  cv::Mat img(h, w, CV_8UC1, originalImage);
  cv::Mat imgColor;
  cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);

  // Cálculo del centro de la imagen.
  int xCenter = (w / 2);
  int yCenter = (h / 2);

  // Vector que almacena las líneas junto con su peso respectivo.
  std::vector<std::pair<cv::Vec2f, int>>linesWithWeights;

  // Iteración para llenar el vector con las líneas halladas.
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

      // Peso a colocar.
      int weight = h_hough[((rIdx * degreeBins) + tIdx)];

      // Push de la línea obtenida.
      if (weight > 0) {
        float localReValue = ((rIdx * rScale) - rMax);
        float theta = (tIdx * radInc);
        linesWithWeights.push_back(std::make_pair(cv::Vec2f(theta, localReValue), weight));
      }
    }
  }

  // Proceso para ordenar las líneas por peso en orden descendente.
  std::sort(
    linesWithWeights.begin(),
    linesWithWeights.end(),
    [](const std::pair<cv::Vec2f, int> &a, const std::pair<cv::Vec2f, int> &b) {
      return a.second > b.second;
    }
  );

  // Ciclo para dibujar las primeras líneas obtenidas.
  for (int i = 0; i < std::min(threshold, static_cast<int>(linesWithWeights.size())); i++) {

    // Valores necesarios para la obtención de la línea.
    cv::Vec2f lineParams = linesWithWeights[i].first;
    float theta = lineParams[0];
    float r = lineParams[1];

    // Coseno y seno del ángulo iterado.
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    // Valores en X y en Y, es decir, punto encontrado.
    double xValue = (xCenter - (r * cosTheta));
    double yValue = (yCenter - (r * sinTheta));
    double alpha = 1000;

    // Creación de la línea con OpenCV.
    cv::line(
      imgColor,
      cv::Point(cvRound(xValue + (alpha * (-sinTheta))),
      cvRound(yValue + (alpha * cosTheta))),
      cv::Point(cvRound(xValue - (alpha * (-sinTheta))),
      cvRound(yValue - (alpha * cosTheta))),
      cv::Scalar(255, 0, 0),
      1,
      cv::LINE_AA
    );
  }

  // Guardado de la imagen con líneas detectadas.
  cv::imwrite(outputFileName, imgColor);
}

// Función main que se encarga de ejecutar el programa.
int main(int argc, char **argv) {

  // Verificación de errores en caso de no pasar una imagen.
  if (argc < 3) {
    printf("Usage: ./hough <pgm-image> <threshold>\n");
    return EXIT_FAILURE;
  }

  // Carga de la imagen pasada como argumento de consola.
  PGMImage inImg(argv[1]);

  // Carga del threshold a utilizar.
  int threshold = strtol(argv[2], NULL, 10);

  // Obtención del ancho y alto de la imagen.
  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // // Instancia de d_Cos y d_Sin a utiliizar.
  // float* d_Cos;
  // float* d_Sin;

  // // Alocación de memoria en la GPU.
  // cudaMalloc((void**) &d_Cos, sizeof(float)* degreeBins);
  // cudaMalloc((void**) &d_Sin, sizeof(float)* degreeBins);

  // Llamada a la versión secuencial del cálculo de la transformada de Hough.
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // Obtención de pcCos, pcSin y los radiantes iniciales.
  float *pcCos = (float*) malloc(sizeof(float) * degreeBins);
  float *pcSin = (float*) malloc(sizeof(float) * degreeBins);
  float rad = 0;

  // Ciclo que obtiene el coseno y seno de los radiantes dados hasta llegar al límite.
  for (int i = 0; i < degreeBins; i++) {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  // Obtención de los valores de r máximos para la versión de CUDA de la transformada.
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = ((2 * rMax) / rBins);

  // Copia de memoria de las matrices a utilizar en memoria constante.
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  // // Copia de memoria de las matrices a utilizar.
  // cudaMemcpy(d_Cos, pcCos, sizeof(float)* degreeBins, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_Sin, pcSin, sizeof(float)* degreeBins, cudaMemcpyHostToDevice);

  // Instancia de los valores a pasar a la versión paralela.
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  // Obtención de los pixeles de la imagen.
  h_in = inImg.pixels;

  // Alocación de memoria para el procedimiento.
  h_hough = (int*) malloc(degreeBins * rBins * sizeof(int));

  // Alocación de memoria en la GPU para las variables a usar.
  cudaMalloc((void **) &d_in, (sizeof(unsigned char) * w * h));
  cudaMalloc((void **) &d_hough, (sizeof(int) * degreeBins * rBins));
  cudaMemcpy(d_in, h_in, (sizeof(unsigned char) * w * h), cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, (sizeof(int) * degreeBins * rBins));

  // Cálculod el número de bloques a utilizar.
  int blockNum = ceil((w * h) / 256);

  // Instancia de los eventos y el tiempo transcurrido.
  cudaEvent_t start, stop;
  float elapsedTime;

  // Crear eventos CUDA.
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Marcar el evento de inicio.
  cudaEventRecord(start, 0);

  // Llamada al kernel del programa.
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

  // Marcar el evento de finalización.
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Calcular el tiempo transcurrido.
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Copia de regreso de los resultados calculados por el GPU.
  cudaMemcpy(h_hough, d_hough, (sizeof(int) * degreeBins * rBins), cudaMemcpyDeviceToHost);

  // Impresión de los valores que difieren entre CPU y GPU.
  for (int i = 0; i < (degreeBins * rBins); i++) {
    if (cpuht[i] != h_hough[i]) {
      printf("Calculation mismatch at: %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
  }

  // Impresión de un mensaje de finalización.
  printf("Done!\n");

  // Imprimir el tiempo transcurrido.
  printf("Tiempo de ejecución del kernel: %f ms\n", elapsedTime);

  // Proceso de dibujar la imagen con las líneas halladas.
  drawAndSaveLines("houghconstante.jpg", inImg.pixels, w, h, h_hough, rScale, rMax, threshold);

  // Liberación de memoria en la GPU.
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  // Liberación del espacio utilizado para el proceso.
  free(h_hough);

  // Liberación de memoria en el CPU.
  delete[] cpuht;

  // Retorno correcto del programa.
  return 0;
}