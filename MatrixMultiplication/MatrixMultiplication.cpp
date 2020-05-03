#include <iostream>
#include <chrono>
#include <CL/cl.hpp>
#include "OpenCLInfo.h"

void mat_mul(const int size, const float* A, const float* B, float* C);
void randomInit(float* data, const int   size);
void print_time(std::chrono::microseconds duration_us);

int main()
{
    
    const int size = 1024;
    float* A = new float[size * size];
    float* B = new float[size * size];
    float* cpu_out = new float[size * size];
    float* gpu_1_out = new float[size * size];
    float* gpu_2_out = new float[size * size];
    float* gpu_3_out = new float[size * size];
    float* gpu_4_out = new float[size * size];
    
    randomInit(A, size * size);
    randomInit(B, size * size);
    for (size_t i = 0; i < size; i++)
    {
        cpu_out[i] = 0;
        gpu_1_out[i] = 0;
        gpu_2_out[i] = 0;
        gpu_3_out[i] = 0;
        gpu_4_out[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    //mat_mul(size, A, B, cpu_out);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on cpu Version 1 took: ";
    print_time(duration_us);

    std::vector<std::string> kernelNames;
    kernelNames.push_back("MatMul.cl");
    OpenCLInfo openCLInfo(kernelNames, "NVIDIA CUDA", CL_DEVICE_TYPE_GPU);
    cl_int err;

    cl::Buffer A_Buffer(openCLInfo.m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, A, NULL);
    cl::Buffer B_Buffer(openCLInfo.m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, B, NULL);
    cl::Buffer gpu1_Buffer(openCLInfo.m_context, CL_MEM_WRITE_ONLY,                    sizeof(float) * size * size, NULL, NULL);
    
    cl::Kernel kernel(openCLInfo.GetProgram(0), "mat_mul", &err);
    int iArg = 0;
    kernel.setArg(iArg++, size);
    kernel.setArg(iArg++, A_Buffer);
    kernel.setArg(iArg++, B_Buffer);
    kernel.setArg(iArg++, gpu1_Buffer);

    // execute kernel
    start = std::chrono::high_resolution_clock::now();

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size, size), cl::NullRange);
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(gpu1_Buffer, CL_TRUE, 0, sizeof(float) * size * size, gpu_1_out);
    
    stop = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on gpu Version 1 took: ";
    print_time(duration_us);


    cl::Buffer gpu2_Buffer(openCLInfo.m_context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, gpu_2_out, NULL);
    cl::Kernel kernel_v2(openCLInfo.GetProgram(0), "mat_mul_v2", &err);
    iArg = 0;
    kernel_v2.setArg(iArg++, size);
    kernel_v2.setArg(iArg++, A_Buffer);
    kernel_v2.setArg(iArg++, B_Buffer);
    kernel_v2.setArg(iArg++, gpu2_Buffer);

    // execute kernel
    start = std::chrono::high_resolution_clock::now();

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, cl::NDRange(size), cl::NullRange);
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(gpu2_Buffer, CL_TRUE, 0, sizeof(float) * size * size, gpu_2_out);

    stop = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on gpu Version 2 took: ";
    print_time(duration_us);


    // execute kernel
    start = std::chrono::high_resolution_clock::now();

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, cl::NDRange(size), cl::NDRange(32));
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(gpu2_Buffer, CL_TRUE, 0, sizeof(float) * size * size, gpu_2_out);

    stop = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on gpu Version 2 took: ";
    print_time(duration_us);

    cl::Buffer gpu3_Buffer(openCLInfo.m_context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, gpu_3_out, NULL);
    cl::Kernel kernel_v3(openCLInfo.GetProgram(0), "mat_mul_v3", &err);
    iArg = 0;
    kernel_v3.setArg(iArg++, size);
    kernel_v3.setArg(iArg++, A_Buffer);
    kernel_v3.setArg(iArg++, B_Buffer);
    kernel_v3.setArg(iArg++, gpu3_Buffer);

    // execute kernel
    start = std::chrono::high_resolution_clock::now();

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel_v3, cl::NullRange, cl::NDRange(size), cl::NDRange(32));
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(gpu3_Buffer, CL_TRUE, 0, sizeof(float) * size * size, gpu_3_out);

    stop = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on gpu Version 3 took: ";
    print_time(duration_us);

    cl::Buffer gpu4_Buffer(openCLInfo.m_context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, gpu_4_out, NULL);
    cl::LocalSpaceArg Btemp = cl::Local(sizeof(float) * size);
    cl::Kernel kernel_v4(openCLInfo.GetProgram(0), "mat_mul_v4", &err);
    iArg = 0;
    kernel_v4.setArg(iArg++, size);
    kernel_v4.setArg(iArg++, A_Buffer);
    kernel_v4.setArg(iArg++, B_Buffer);
    kernel_v4.setArg(iArg++, gpu4_Buffer);
    kernel_v4.setArg(iArg++, Btemp);

    // execute kernel
    start = std::chrono::high_resolution_clock::now();

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, cl::NDRange(size), cl::NDRange(32));
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(gpu4_Buffer, CL_TRUE, 0, sizeof(float) * size * size, gpu_4_out);

    stop = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Matrix Multiplication 1024x1024 on gpu Version 4 took: ";
    print_time(duration_us);


    std::cout << "\n\n\n                            Check output from gpu against cpu" << std::endl;
    for (size_t i = 0; i < 5; ++i)
    {
        std::cout << "Element #" << i << ": cpu_out: " << cpu_out[i] << " gpu_1_out: " << gpu_1_out[i] << " gpu_2_out: " << gpu_2_out[i] << " gpu_3_out: " << gpu_3_out[i] << " gpu_4_out: " << gpu_4_out[i] << std::endl;
    }
    std::cout << "\n\n" << std::endl;
    for (size_t i = size - 1; i > size - 6; --i)
    {
        std::cout << "Element #" << i<<": cpu_out: " << cpu_out[i] << " gpu_1_out: " << gpu_1_out[i] << " gpu_2_out: " << gpu_2_out[i] << " gpu_3_out: " << gpu_3_out[i] << " gpu_4_out: " << gpu_4_out[i] << std::endl;
    }


}



void randomInit(float* data, const int   size) {
    for (int i = 0; i < size; ++i)
        data[i] = ((double)rand() / (RAND_MAX)) + 1;
}

void mat_mul(const int  size, const float* A, const float* B, float* C)
{
    for (int i = 0; i < size; ++i) {
        for (int  j = 0; j < size; ++j) {
            for (int  k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void print_time(std::chrono::microseconds duration_us) {
    std::cout << int(duration_us.count() / 1000000) << " s "
              << int(duration_us.count() / 1000) - int(duration_us.count() / 1000000) * 1000 << " ms "
              << duration_us.count() - int(duration_us.count() / 1000) * 1000<< " us "
              << std::endl;
}