
__kernel void mat_mul(const int  size, __global float* A, __global float* B, __global float* C)
{
	int  i = get_global_id(0);
	int  j = get_global_id(1);
	for (int  k = 0; k < size; k++) {
		// C(i, j) = sum(over k) A(i,k) * B(k,j)
		C[i * size + j] += A[i * size + k] * B[k * size + j];
	}
}

__kernel void mat_mul_v2(const int  size, __global float* A, __global float* B, __global float* C)
{
	int  i = get_global_id(0);
	float tmp;
	for (int j = 0; j < size; ++j) {
		tmp = 0.0f;
		for (int k = 0; k < size; ++k) {
			tmp += A[i * size + k] * B[k * size + j]; 
		}
		C[i * size + j] = tmp;
	}
}

__kernel void mat_mul_v3(const int  size, __global float* A, __global float* B, __global float* C)
{
	int  i = get_global_id(0);
	float tmp;
	float Atemp[1024];
	for (int k = 0; k < size; k++)
		Atemp[k] = A[i * size + k];

	for (int j = 0; j < size; ++j) {
		tmp = 0.0f;
		for (int k = 0; k < size; ++k) {
			tmp += Atemp[k] * B[k * size + j];
		}
		C[i * size + j] = tmp;
	}
}

__kernel void mat_mul_v4(const int  size, __global float* A, __global float* B, __global float* C, __local float* Btemp)
{
	int  i = get_global_id(0);
	int iloc = get_local_id(0);
	int nloc = get_local_size(0);

	float tmp;
	float Atemp[1024];

	for (int k = 0; k < size; k++) {
		Atemp[k] = A[i * size + k];
	}

	for (int j = 0; j < size; ++j) {
		for (int k = iloc; k < size; k += nloc) {
			Btemp[k] = B[k * size + j];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		tmp = 0.0f;
		for (int k = 0; k < size; ++k) {
			tmp += Atemp[k] * Btemp[k];
		}
		C[i * size + j] = tmp;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}