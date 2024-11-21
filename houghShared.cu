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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include "common/bmp.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float D_COS[degreeBins];
__constant__ float D_SIN[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // Cálculo y verificación de que el ID sea válido.
  int localID = threadIdx.x;
  int gloID = localID + blockIdx.x * blockDim.x;

  // Instancia de la memoria compartida para el acumulador local.
  __shared__ int localAcc[(degreeBins * rBins)];

  // Ciclo que inicia el acumulador local.
  for (int i = localID; i < (degreeBins * rBins); i += blockDim.x) {
    localAcc[i] = 0;
  }

  // Sincronización de threads.
  __syncthreads();

  // Verificación de que el ID global esté dentro de la imagen.
  if (gloID < (w * h)) {

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
        float r = ((xCoord * D_COS[tIdx]) + (yCoord * D_SIN[tIdx]));
        int rIdx = (int)((r + rMax) / rScale);
        if ((rIdx >= 0) && (rIdx < rBins)) {
          atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
      }
    }
  }

  // Sincronización de threads.
  __syncthreads();

  // Suma final de los valores del acumulador local usando un loop.
  for (int i = localID; i < (degreeBins * rBins); i += blockDim.x) {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

// Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
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
      float r = ((xCoord * D_COS[tIdx]) + (yCoord * D_SIN[tIdx]));
      int rIdx = ((r + rMax) / rScale);

      // Barrera para sincronizar los hilos dentro del bloque.
      __syncthreads();

      // Actualización del acumulador.
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Calculo del thread global
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

void get_bmp_pixels(Color *pixels, int w, int h, PGMImage &inImg, int *acc, int threshold, float rMax, float rScale, float *pcCos, float *pcSin) {
  // Inicializa la imagen BMP con los colores de la imagen PGM
  for (int i = 0; i < w * h; i++) {
    if (inImg.pixels[i] > 5) {
      pixels[(w * h) - i - 1] = {255, 255, 255}; // Blanco
    } else {
      pixels[(w * h) - i - 1] = {0, 0, 0}; // Negro
    }
  }

  // Dibuja las líneas encontradas en la imagen BMP
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      if (acc[rIdx * degreeBins + tIdx] > threshold) {
        float r = rIdx * rScale - rMax;
        float cosTheta = pcCos[tIdx];
        float sinTheta = pcSin[tIdx];

        for (int x = 0; x < w; x++) {
          int y = (r - (x - w / 2) * cosTheta) / sinTheta + h / 2;
          if (y >= 0 && y < h) {
            pixels[y * w + x] = {0, 255, 0}; // Verde
          }
        }

        for (int y = 0; y < h; y++) {
          int x = (r - (y - h / 2) * sinTheta) / cosTheta + w / 2;
          if (x >= 0 && x < w) {
            pixels[y * w + x] = {0, 255, 0}; // Verde
          }
        }
      }
    }
  }
}

int Exec_Global(PGMImage &inImg) {
  int i;

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);

  // Tomar el tiempo con CUDA Events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  printf("Done! Tiempo de llamada al kernel con memoria global: %f milisegundos\n", miliseconds);

  // Llena un nuevo array con los pixeles que forman las lineas encontradas
  Color pixels[w * h];

  // Calcula promedio y varianza para determinar el threshold
  float avg = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    avg += h_hough[i];
  
  avg /= degreeBins * rBins;

  float var = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    var += (h_hough[i] - avg) * (h_hough[i] - avg);

  var /= degreeBins * rBins;

  int threshold = (int) avg + 2*sqrt(var);
  get_bmp_pixels(pixels, w, h, inImg, h_hough, threshold, rMax, rScale, pcCos, pcSin);

  // Guarda la imagen BMP
  BMPImage outImg(pixels, w, h);
  outImg.save("global.bmp");

  // Liberar memoria
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  free(pcCos);
  free(pcSin);

  free(h_hough);

  cudaFree(d_hough);
  cudaFree(d_in);

  return 0;
}

int Exec_Constant(PGMImage &inImg) {
  int i;

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  cudaMalloc ((void **) &D_COS, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &D_SIN, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpyToSymbol(D_COS, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(D_SIN, pcSin, sizeof (float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  int blockNum = ceil (w * h / 256);

  // Tomar el tiempo con CUDA Events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start);
  GPU_HoughTranConst <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  cudaEventRecord(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  printf("Done! Tiempo de llamada al kernel con memoria constante: %f milisegundos\n", miliseconds);

  // Llena un nuevo array con los pixeles que forman las lineas encontradas
  Color pixels[w * h];

  // Calcula promedio y varianza para determinar el threshold
  float avg = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    avg += h_hough[i];
  
  avg /= degreeBins * rBins;

  float var = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    var += (h_hough[i] - avg) * (h_hough[i] - avg);

  var /= degreeBins * rBins;

  int threshold = (int) avg + 2*sqrt(var);
  get_bmp_pixels(pixels, w, h, inImg, h_hough, threshold, rMax, rScale, pcCos, pcSin);

  // Guarda la imagen BMP
  BMPImage outImg(pixels, w, h);
  outImg.save("constant.bmp");

  // Liberar memoria
  cudaFree(D_COS);
  cudaFree(D_SIN);

  free(pcCos);
  free(pcSin);

  free(h_hough);

  cudaFree(d_hough);
  cudaFree(d_in);

  return 0;
}

int Exec_Shared(PGMImage &inImg) {
  int i;

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  cudaMalloc ((void **) &D_COS, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &D_SIN, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpyToSymbol(D_COS, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(D_SIN, pcSin, sizeof (float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  int blockNum = ceil (w * h / 256);

  // Tomar el tiempo con CUDA Events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start);
  GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  cudaEventRecord(stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  printf("Done! Tiempo de llamada al kernel con memoria compartida: %f milisegundos\n", miliseconds);

  // Llena un nuevo array con los pixeles que forman las lineas encontradas
  Color pixels[w * h];

  // Calcula promedio y varianza para determinar el threshold
  float avg = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    avg += h_hough[i];
  
  avg /= degreeBins * rBins;

  float var = 0;
  for (i = 0; i < degreeBins * rBins; i++)
    var += (h_hough[i] - avg) * (h_hough[i] - avg);

  var /= degreeBins * rBins;

  int threshold = (int) avg + 2*sqrt(var);
  get_bmp_pixels(pixels, w, h, inImg, h_hough, threshold, rMax, rScale, pcCos, pcSin);

  // Guarda la imagen BMP
  BMPImage outImg(pixels, w, h);
  outImg.save("shared.bmp");

  // Liberar memoria
  cudaFree(D_COS);
  cudaFree(D_SIN);

  free(pcCos);
  free(pcSin);

  free(h_hough);

  cudaFree(d_hough);
  cudaFree(d_in);

  return 0;
}

//*****************************************************************
int main (int argc, char **argv)
{

  PGMImage inImg (argv[1]);

  Exec_Global(inImg);
  Exec_Constant(inImg);
  Exec_Shared(inImg);

  return 0;
}
