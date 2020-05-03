#include <iostream>
#include <CL/cl.hpp>
#include "OpenCLInfo.h"

int main() {
	std::vector<std::string> kernelNames;
	kernelNames.push_back("HelloWorld.cl");
    kernelNames.push_back("Vectors.cl");
	OpenCLInfo openCLInfo(kernelNames, "NVIDIA CUDA", CL_DEVICE_TYPE_GPU);
    cl_int err;

    char buf[20];
    cl::Buffer memBuf(openCLInfo.m_context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(openCLInfo.GetProgram(0), "HelloWorld", &err);
    kernel.setArg(0, memBuf);


    openCLInfo.m_queue.enqueueTask(kernel);
    openCLInfo.m_queue.finish();
    openCLInfo.m_queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;

    cl::Buffer memBuf2(openCLInfo.m_context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel2(openCLInfo.GetProgram(0), "HelloEveryOne", &err);
    kernel2.setArg(0, memBuf2);

    openCLInfo.m_queue.enqueueTask(kernel2);
    openCLInfo.m_queue.enqueueReadBuffer(memBuf2, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;

    const int size = 10;
    int* A = new int[size];
    int* B = new int[size];
    int* C = new int[size];

    for (size_t i = 0; i < size; i++)
    {
        A[i] = i;
        B[i] = size - 2*i;
        C[i] = 0;
    }

    cl::Buffer A_Buffer(openCLInfo.m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, A, NULL);
    cl::Buffer B_Buffer(openCLInfo.m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * size, B, NULL);
    cl::Buffer C_Buffer(openCLInfo.m_context, CL_MEM_WRITE_ONLY,                       sizeof(int) * size, NULL, NULL);
    cl::Kernel kernel_3(openCLInfo.GetProgram(1), "vector_add", &err);

    int iArg = 0;
    kernel_3.setArg(iArg++, A_Buffer);
    kernel_3.setArg(iArg++, B_Buffer);
    kernel_3.setArg(iArg++, C_Buffer);

    openCLInfo.m_queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(size), cl::NullRange);
    openCLInfo.m_queue.finish();
    // read output array
    openCLInfo.m_queue.enqueueReadBuffer(C_Buffer, CL_TRUE, 0, sizeof(int) * size, C);

    for (size_t i = 0; i < size; i++)
    {
        std::cout << A[i] << "+" << B[i] << "=" << C[i] << std::endl;;
    }

	return 0;
}
