#include "OpenCLInfo.h"
#include <iostream>
#include <fstream>

OpenCLInfo::OpenCLInfo(const std::vector<std::string>& kernelsSources, const std::string& preferredPlatformName, const cl_device_type deviceType) {
	// error variable to check for opencl errors during initialization 
	cl_int err = CL_SUCCESS;
	// get different platforms IDs;
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	// flag to indicate if the preffered platform was found or no through the next loop
	bool platform_found = false;
	// Loop through the platforms id to check for the preffered platform
	std::vector<int>::size_type platforms_count = platforms.size();
	for (std::vector<int>::size_type i = 0; i != platforms_count; ++i) {
		// variable to be used to hold retrieved platform name
		std::string platformName;
		// get name of ith platform
		platforms[i].getInfo(CL_PLATFORM_NAME, &platformName);
		// remove string terminator from platformName if exists.
		if (platformName.back() == '\0') platformName.pop_back();
		// Check for ith platform if it's the prefferd one or no.
		// if yes, stored it change flag to true then break out of the current loop
		if (!preferredPlatformName.compare(platformName)) {
		
			m_platform = platforms[i];
			platform_found = true; // change flag to true to indicate that the preffered platform was found.
			break;
		}
	}

	if (!platform_found) std::cerr << "Error: Preffered platform was not found, Please check the given name again!" << std::endl;

	// get different devices IDs for our platform;
	std::vector<cl::Device> devices;
	err = m_platform.getDevices(deviceType, &devices);

	if (err != CL_SUCCESS) std::cerr << "Error: Error in getting devices IDs, ErrorType: " + std::to_string(err) << std::endl;
	
	m_device = devices[0]; // stores the first device found of type deviceType, for no reason :"D

	m_context = cl::Context(m_device); // initialize context for the choosen device.

	m_queue = cl::CommandQueue(m_context, m_device); // initialize queue for our context.

	std::vector<int>::size_type kernels_count = kernelsSources.size();
	for (std::vector<int>::size_type i = 0; i != kernels_count; ++i) {
		// open the ith kernel source file.
		std::ifstream sourceFile(kernelsSources[i]);
		// read its content into string and cread cl sources varaible out of it.
		std::string src(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
		// creating a program for the source code inside our context.
		cl::Program program(m_context, sources);
		// build the kernel into opencl binary.
		err = program.build("-cl-std=CL1.2");

		if (err != CL_SUCCESS)  std::cerr << "Error: Error in building kernel: " + kernelsSources[i] + ", ErrorType: " + std::to_string(err) << std::endl;
		// add the new built program to our programs vector
		m_programs.push_back(program);
	}
}

cl::Program OpenCLInfo::GetProgram(size_t i) {
	if (i >= m_programs.size())
		std::cerr << "Error: Trying to access program outside the range of programs vector. In: OpenCLInfo::GetProgram(size_t i)" << std::endl;
	else
		return m_programs[i];
}