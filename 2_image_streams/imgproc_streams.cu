/*
 *     
 *  IMAGE PROCESSING
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"

#define pixel(i, j, n)  (((j)*(n)) +(i))
int B = 16;


/*read*/
void  readimg(char * filename,int nx, int ny, int * image){
  
   FILE *fp=NULL;

   fp = fopen(filename,"r");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fscanf(fp,"%d", &image[pixel(i,j,nx)]);      
      }
   }
   fclose(fp);
}

/* save */   
void saveimg(char *filename,int nx,int ny,int *image){

   FILE *fp=NULL;
   fp = fopen(filename,"w");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fprintf(fp,"%d ", image[pixel(i,j,nx)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/*invert*/
__global__ void invert(int* image, int* image_invert, int nx, int ny){

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indx >= 0 && indx <= nx - 1){
      if(indy >= 0 && indy <= ny - 1){
         image_invert[pixel(indx,indy,nx)] = 255-image[pixel(indx,indy,nx)];
      }
   }
}

/*smooth*/
__global__ void smooth(int* image, int* image_smooth, int nx, int ny){

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indx >= 0 && indx <= nx - 1){
      if(indy >= 0 && indy <= ny - 1){
         
         if(indx == 0 || indx == nx-1){
            image_smooth[pixel(indx,indy,nx)] = 0;
         }
         else if(indy == 0 || indy == ny-1){
            image_smooth[pixel(indx,indy,nx)] = 0;
         }
         else{
            image_smooth[pixel(indx,indy,nx)] = (image[pixel(indx-1,indy+1,nx)]+image[pixel(indx,indy+1,nx)]+image[pixel(indx+1,indy+1,nx)]+image[pixel(indx-1,indy,nx)]+image[pixel(indx,indy,nx)]+image[pixel(indx+1,indy,nx)]+image[pixel(indx-1,indy-1,nx)]+image[pixel(indx,indy-1,nx)]+image[pixel(indx+1,indy-1,nx)])/9;
            
            if(image_smooth[pixel(indx, indy, nx)] < 0){
            image_smooth[pixel(indx, indy, nx)] = 0;
            } 

            else if (image_smooth[pixel(indx, indy, nx)] > 255){
               image_smooth[pixel(indx, indy, nx)] = 255;
            }
         }
      }
   }
}

/*detect*/
__global__ void detect(int* image, int* image_detect, int nx, int ny){

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indx >= 0 && indx <= nx - 1){
      if(indy >= 0 && indy <= ny - 1){

         int pixel_check = image[pixel(indx-1,indy,nx)]+image[pixel(indx+1,indy,nx)]+image[pixel(indx,indy-1,nx)]+image[pixel(indx, indy+1, nx)]-(4*image[pixel(indx,indy,nx)]);
         
         if(indx == 0 || indx == nx-1){
            image_detect[pixel(indx,indy,nx)] = 0;
         }
         else if(indy == 0 || indy == ny-1){
            image_detect[pixel(indx,indy,nx)] = 0;
         }
         else if(pixel_check > 255){
            image_detect[pixel(indx,indy,nx)] = 255;
         }
         else if(pixel_check < 0){
            image_detect[pixel(indx,indy,nx)] = 0;
         }
         else{
         image_detect[pixel(indx,indy,nx)] = pixel_check;
        }
      }
   }
}

/*enhance*/
__global__ void enhance(int* image,int *image_enhance,int nx, int ny){
      
   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indx >= 0 && indx <= nx - 1){
      if(indy >= 0 && indy <= ny - 1){

         int pixel_check = (5*image[pixel(indx,indy,nx)]) - (image[pixel(indx-1,indy,nx)] + image[pixel(indx+1,indy,nx)] + image[pixel(indx,indy-1,nx)] + image[pixel(indx,indy+1,nx)]);
         
         if(indx == 0 || indx == nx-1){
            image_enhance[pixel(indx,indy,nx)] = 0;
         }
         else if(indy == 0 || indy == ny-1){
            image_enhance[pixel(indx,indy,nx)] = 0;
         }
         else if(pixel_check > 255){
            image_enhance[pixel(indx,indy,nx)] = 255;
         }
         else if(pixel_check < 0){
            image_enhance[pixel(indx,indy,nx)] = 0;
         }
         else{
            image_enhance[pixel(indx,indy,nx)] = pixel_check;
         }
      }
   }
}

/* Main program */
int main (int argc, char *argv[])
{
   int    nx,ny;
   char   filename[250];
   float runtime = 0;

   /*---- cuda timing ----*/
   //double *dev_blockMax;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   /* Get parameters */
   if (argc != 4) 
   {
      printf ("Usage: %s image_name N M \n", argv[0]);
      exit (1);
   }
   sprintf(filename, "%s.txt", argv[1]);
   nx  = atoi(argv[2]);
   ny  = atoi(argv[3]);

   printf("%s %d %d\n", filename, nx, ny);

   /* Saving blocks */
   dim3 dimBlock(B,B,1);
   int dimgx = (nx+B-1)/B;
   int dimgy = (ny+B-1)/B;
   dim3 dimGrid(dimgx, dimgy, 1);

   // cuda stream
   cudaStream_t stream1, stream2, stream3, stream4;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   cudaStreamCreate(&stream3);
   cudaStreamCreate(&stream4);

   /* for cuda mallocs Allocate*/
   int* dev_image;
   int* dev_image_invert;
   int* dev_image_smooth;
   int* dev_image_detect;
   int* dev_image_enhance;

   /* Allocate CPU and GPU pointers */
   //Para pinned memory
   int*   image;
   cudaMallocHost((void **) &image, nx*ny*sizeof(int));

   int*   image_invert;
   cudaMallocHost((void **) &image_invert, nx*ny*sizeof(int));

   int*   image_smooth;
   cudaMallocHost((void **) &image_smooth, sizeof(int)*nx*ny);  

   int*   image_detect;
   cudaMallocHost((void **) &image_detect, sizeof(int)*nx*ny);  

   int*   image_enhance; 
   cudaMallocHost((void **) &image_enhance, sizeof(int)*nx*ny);

   /*
   sin pinned memory
   int*   image =(int *) malloc(sizeof(int)*nx*ny);
   int*   image_invert  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_smooth  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_detect = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_enhance = (int *) malloc(sizeof(int)*nx*ny); 
   */

   cudaMalloc((void **)&dev_image, sizeof(int)*nx*ny);
   cudaMalloc((void **)&dev_image_invert, sizeof(int)*nx*ny);
   cudaMalloc((void **)&dev_image_smooth, sizeof(int)*nx*ny);
   cudaMalloc((void **)&dev_image_detect, sizeof(int)*nx*ny);
   cudaMalloc((void **)&dev_image_enhance, sizeof(int)*nx*ny);

   /* Read image and save in array imgage */
   readimg(filename,nx,ny,image);
   cudaMemcpy(dev_image, image, nx*ny*sizeof(int), cudaMemcpyHostToDevice);

   /* llamar funciones con stream 1 y el 0 ese en el kernel. */

   cudaEventRecord(start);
   invert<<<dimGrid,dimBlock, 0, stream1>>>(dev_image, dev_image_invert, nx, ny);
   
   smooth<<<dimGrid, dimBlock, 0, stream2>>>(dev_image, dev_image_smooth, nx, ny);
   
   detect<<<dimGrid, dimBlock, 0, stream3>>>(dev_image, dev_image_detect, nx, ny);
   
   enhance<<<dimGrid, dimBlock, 0, stream4>>>(dev_image, dev_image_enhance, nx, ny);
   cudaEventRecord(stop);

   cudaMemcpyAsync(image_invert, dev_image_invert, nx*ny*sizeof(int), cudaMemcpyDeviceToHost, stream1);
   cudaStreamSynchronize(stream1);

   cudaMemcpyAsync(image_smooth, dev_image_smooth, sizeof(int)*nx*ny, cudaMemcpyDeviceToHost, stream2);
   cudaStreamSynchronize(stream2);

   cudaMemcpyAsync(image_detect, dev_image_detect, sizeof(int)*nx*ny, cudaMemcpyDeviceToHost, stream3);
   cudaStreamSynchronize(stream3);

   cudaMemcpyAsync(image_enhance, dev_image_enhance, sizeof(int)*nx*ny, cudaMemcpyDeviceToHost, stream4);
   cudaStreamSynchronize(stream4);

   cudaEventSynchronize(stop);
   
  /* Print runtime */
   cudaEventElapsedTime(&runtime, start, stop);
   printf("Total time transformations: %f\n",runtime);
   
   /* Save images */
   char fileout[255]={0};
   sprintf(fileout, "%s-inverse.txt", argv[1]);
   saveimg(fileout,nx,ny,image_invert);
   sprintf(fileout, "%s-smooth.txt", argv[1]);
   saveimg(fileout,nx,ny,image_smooth);
   sprintf(fileout, "%s-detect.txt", argv[1]);
   saveimg(fileout,nx,ny,image_detect);
   sprintf(fileout, "%s-enhance.txt", argv[1]);
   saveimg(fileout,nx,ny,image_enhance);

   /* Deallocate CPU and GPU pointers*/
   cudaFreeHost(image);
   cudaFreeHost(image_invert);
   cudaFreeHost(image_smooth);
   cudaFreeHost(image_detect);
   cudaFreeHost(image_enhance);

   cudaFree(dev_image);
   cudaFree(dev_image_invert);
   cudaFree(dev_image_detect);
   cudaFree(dev_image_smooth);
   cudaFree(dev_image_enhance);

   cudaStreamDestroy(stream1);
   cudaStreamDestroy(stream2);
   cudaStreamDestroy(stream3);
   cudaStreamDestroy(stream4);
}