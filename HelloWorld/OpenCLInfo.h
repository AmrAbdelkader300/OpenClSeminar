#include <CL/cl.hpp>

class OpenCLInfo
{
public:
	OpenCLInfo() = default;

    /**
     * \breif Constructor for OpenCLInfo class which is used to initialize opencl parameters:
     *        platform, device, context and commaned queue using the following parameters.
     *		  the selected device would be the first one of its type in the platform.
     * \param kernelsSources         [in] a vector of strings which contains kernels source files name
     * \param preferredPlatformName  [in] a string which holds the preferred platoform name, ex: NVIDIA CUDA, Intel(R) OpenCL
     * \param deviceType             [in] opencl enum used to select one of the devices in the platform. ex: GPU, CPU, Accelators
     *                                    possible values: CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR
     *                                                     CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_ALL
     */
	OpenCLInfo(const std::vector<std::string>& kernelsSources, const std::string& preferredPlatformName, const cl_device_type deviceType);

    /**
     * \breif return the ith program in m_programs list.
     * \param i [int] index of required program.
     */
    cl::Program GetProgram(size_t i);

	cl::Context m_context;
	cl::CommandQueue m_queue;
	std::vector<cl::Program> m_programs;

private:
    cl::Platform m_platform;
    cl::Device m_device;
};

