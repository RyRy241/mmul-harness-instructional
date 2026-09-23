#include <cstring>
#include <vector>
const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   const int bs = block_size;

   std::vector<double> Ablk(bs*bs), Bblk(bs*bs), Cblk(bs*bs);

   for (int ii = 0; ii < n; ii += bs) {
      for (int jj = 0; jj < n; jj += bs) {

         for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < bs; ++j) {
               Cblk[i*bs + j] = C[(ii+i)*n + (jj+j)];
            }
         }

         for (int kk = 0; kk < n; kk += bs) {

            for (int i = 0; i < bs; ++i) {
               for (int k = 0; k < bs; ++k) {
                  Ablk[i*bs + k] = A[(ii+i)*n + (kk+k)];
               }
            }

            for (int i = 0; i < bs; ++i) {
               for (int j = 0; j < bs; ++j) {
                  Bblk[i*bs + j] = B[(kk+i)*n + (jj+j)];
               }
            }

            for (int i = 0; i < bs; ++i) {
               for (int k = 0; k < bs; ++k) {
                  for (int j = 0; j < bs; ++j) {
                     Cblk[i*bs + j] += Ablk[i*bs + k] * Bblk[k*bs + j];
                  }
               }
            }
         }

         for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < bs; ++j) {
               C[(ii+i)*n + (jj+j)] = Cblk[i*bs + j];
            }
         }
      }
   }
}
