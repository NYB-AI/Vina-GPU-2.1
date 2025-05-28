#include <wrapcl.h>
#include <iostream>

//https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* getErrorString(cl_int error)
{
    switch (error) {
        // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void checkErr(cl_int err) {
    if (CL_SUCCESS != err) {
        std::cout << "\nErr" << err << ":" << getErrorString(err) << std::endl;
        exit(-1);
    }
    else {
        //printf("Success!");
        //printf(" - ");
    }
    //assert(err == CL_SUCCESS);
    fflush(stdout);
}


void read_file(char** program_file, size_t* program_size, std::string file_path) {
    const char* file_path_trans = file_path.data();
    size_t size;
    char* str;
    std::fstream f(file_path, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        *program_file = str;
        //delete[] str;
        *program_size = size+1;
        return;
    }
   printf("Error: failed to open file\n: %s", file_path.data());
}


void read_n_file(char** program_file, size_t* program_size, std::string file_paths[], size_t num_file) {
    for (int i = 0; i < num_file; i++) {
        read_file(&(program_file[i]), &(program_size[i]), file_paths[i]);
    }
}


void SetupPlatform(cl_platform_id** platforms, cl_int* gpu_platform) {
    *gpu_platform = -1;
    cl_int err;
    cl_uint num_platform;
    size_t size;
    std::string nvidia = "NVIDIA";
    std::string amd = "AMD";
    std::string intel_cpu = "Intel(R) OpenCL";
    std::string intel_graphic_gpu = "Intel(R) OpenCL HD Graphics";
    err = clGetPlatformIDs(0, NULL, &num_platform); checkErr(err);
    *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * (num_platform));
    err = clGetPlatformIDs(num_platform, *platforms, NULL); checkErr(err);
#ifdef NVIDIA_PLATFORM
        int best_platform = -1;
        cl_uint max_devices = 0;
        
        for (int i = 0; i < num_platform; i++) {
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
            char* platform_name = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
            std::string tmp = platform_name;
            
            if (tmp.find(nvidia) != std::string::npos) {
                // Count how many devices this platform has
                cl_uint num_devices = 0;
                err = clGetDeviceIDs((*platforms)[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
                
                if (err == CL_DEVICE_NOT_FOUND) {
                    free(platform_name);
                    continue; // No devices on this platform
                }
                
                checkErr(err);
                
                // If this platform has more devices, select it
                if (num_devices > max_devices) {
                    max_devices = num_devices;
                    best_platform = i;
                }
                
                // Print platform info
                printf("\nFound NVIDIA Platform %d: %s with %d device(s)", i, platform_name, num_devices);
            }
            
            free(platform_name);
        }
        
        if (best_platform != -1) {
            *gpu_platform = best_platform;
            
            // Print the selected platform name
            err = clGetPlatformInfo((*platforms)[best_platform], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
            char* platform_name = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[best_platform], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
            
            printf("\nSelected GPU Platform %d: %s with %d device(s)", best_platform, platform_name, max_devices);
            free(platform_name);
        } else {
            printf("\nCannot find any NVIDIA platform\n");fflush(stdout);
            exit(-1);
        }
#elif AMD_PLATFORM
        for (int i = 0; i < num_platform; i++) {
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
            char* platform_name = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
            std::string tmp = platform_name;
            if (tmp.find(amd) != std::string::npos) {
                *gpu_platform = i;
                printf("GPU Platform: %s", platform_name);fflush(stdout);
#ifdef DISPLAY_ADDITION_INFO
                err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, 0, NULL, &size); checkErr(err);
                char* platform_version = (char*)malloc(sizeof(char) * size);
                err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, size, platform_version, NULL); checkErr(err);
                printf("GPU Platform %d version: %s\n", i, platform_version);
                free(platform_version);
#endif
                break;
            }
        }
        if(*gpu_platform==-1){
            printf("Cannot find any AMD platform\n");fflush(stdout);
            exit(-1);
        }
#elif INTEL_CPU_PLATFORM
    for (int i = 0; i < num_platform; i++) {
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
        char* platform_name = (char*)malloc(sizeof(char) * size);
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
        std::string tmp = platform_name;
        
        if (tmp.find(intel_cpu) != std::string::npos && tmp.find(intel_graphic_gpu) == std::string::npos) {
            *gpu_platform = i;
            printf("GPU Platform: %s", platform_name); fflush(stdout);
#ifdef DISPLAY_ADDITION_INFO
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, 0, NULL, &size); checkErr(err);
            char* platform_version = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, size, platform_version, NULL); checkErr(err);
            printf("GPU Platform %d version: %s\n", i, platform_version);
            free(platform_version);
#endif
            break;
        }
}
    if (*gpu_platform == -1) {
        printf("Cannot find any Intel(R) CPU platform\n"); fflush(stdout);
        exit(-1);
    }
#elif INTEL_GPU_PLATFORM
    for (int i = 0; i < num_platform; i++) {
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
        char* platform_name = (char*)malloc(sizeof(char) * size);
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
        std::string tmp = platform_name;
        if (tmp.find(intel_graphic_gpu) != std::string::npos) {
            *gpu_platform = i;
            printf("GPU Platform: %s", platform_name); fflush(stdout);
#ifdef DISPLAY_ADDITION_INFO
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, 0, NULL, &size); checkErr(err);
            char* platform_version = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, size, platform_version, NULL); checkErr(err);
            printf("GPU Platform %d version: %s\n", i, platform_version);
            free(platform_version);
#endif
            break;
        }
    }
    if (*gpu_platform == -1) {
        printf("Cannot find any Intel(R) GPU platform\n"); fflush(stdout);
        exit(-1);
    }
#else
    for (int i = 0; i < num_platform; i++) {
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, 0, NULL, &size); checkErr(err);
        char* platform_name = (char*)malloc(sizeof(char) * size);
        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_NAME, size, platform_name, NULL); checkErr(err);
        std::string tmp = platform_name;
        if (tmp.find(nvidia) != std::string::npos            || tmp.find(amd) != std::string::npos ||
            tmp.find(intel_graphic_gpu) != std::string::npos || tmp.find(intel_cpu) != std::string::npos) {
            *gpu_platform = i;
            printf("GPU OpenCL Platform: %s", platform_name);fflush(stdout);
#ifdef DISPLAY_ADDITION_INFO
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, 0, NULL, &size); checkErr(err);
            char* platform_version = (char*)malloc(sizeof(char) * size);
            err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, size, platform_version, NULL); checkErr(err);
            printf("GPU Platform %d version: %s\n", i, platform_version);
            free(platform_version);
#endif
            break;
        }
    }
    if(*gpu_platform==-1){
            printf("Cannot find any platform\n");fflush(stdout);
            exit(-1);
    }
#endif
 ////#ifdef DISPLAY_ADDITION_INFO
 //        printf("\nPlatform %d : %s\n", i, platform_name);
 //        free(platform_name);
 //        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, 0, NULL, &size); checkErr(err);
 //        char* platform_version = (char*)malloc(sizeof(char) * size);
 //        err = clGetPlatformInfo((*platforms)[i], CL_PLATFORM_VERSION, size, platform_version, NULL); checkErr(err);
 //        printf("\nPlatform %d version: %s\n", i, platform_version);
 //        free(platform_version);
 ////#endif
    

}


void SetupDevice(cl_platform_id* platforms, cl_device_id** devices, cl_int gpu_platform) {
    cl_uint num_device;
    cl_int err;
    size_t device_name_size;
    cl_ulong mem_size;
    cl_int N = gpu_platform;
    
    // Run a diagnostic on all platforms to check for GPUs
    cl_uint num_platform;
    err = clGetPlatformIDs(0, NULL, &num_platform); checkErr(err);
    
    printf("\n*** NVIDIA GPU DIAGNOSTIC ***");
    int total_gpus = 0;
    
    for (cl_uint i = 0; i < num_platform; i++) {
        size_t platform_name_size;
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_size); checkErr(err);
        char* platform_name = (char*)malloc(sizeof(char) * platform_name_size);
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_size, platform_name, NULL); checkErr(err);
        
        cl_uint plat_num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &plat_num_devices);
        
        if (err == CL_DEVICE_NOT_FOUND) {
            printf("\nPlatform %d: %s - No GPU devices", i, platform_name);
            free(platform_name);
            continue;
        }
        
        checkErr(err);
        
        cl_device_id* plat_devices = (cl_device_id*)malloc(sizeof(cl_device_id) * plat_num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, plat_num_devices, plat_devices, NULL); checkErr(err);
        
        printf("\nPlatform %d: %s - %d GPU device(s)", i, platform_name, plat_num_devices);
        
        for (cl_uint j = 0; j < plat_num_devices; j++) {
            size_t device_name_len;
            err = clGetDeviceInfo(plat_devices[j], CL_DEVICE_NAME, 0, NULL, &device_name_len); checkErr(err);
            char* plat_device_name = (char*)malloc(sizeof(char) * device_name_len);
            err = clGetDeviceInfo(plat_devices[j], CL_DEVICE_NAME, device_name_len, plat_device_name, NULL); checkErr(err);
            
            cl_ulong device_mem;
            err = clGetDeviceInfo(plat_devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_mem, NULL); checkErr(err);
            
            printf("\n  GPU %d.%d: %s (%.1f GB)", i, j, plat_device_name, (double)device_mem/1000000000);
            free(plat_device_name);
            total_gpus++;
        }
        
        free(plat_devices);
        free(platform_name);
    }
    
    printf("\n\nTotal GPUs found: %d", total_gpus);
    printf("\nNOTE: You might see fewer GPUs than physically installed due to OpenCL driver limitations.");
    printf("\n*** END DIAGNOSTIC ***\n");
    
    // Continue with the original function to get devices from the selected platform
#ifdef INTEL_CPU_PLATFORM
    err = clGetDeviceIDs(platforms[N], CL_DEVICE_TYPE_CPU, 0, NULL, &num_device); checkErr(err);
    *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_device);
    err = clGetDeviceIDs(platforms[N], CL_DEVICE_TYPE_CPU, num_device, *devices, NULL); checkErr(err);
#else
    err = clGetDeviceIDs(platforms[N], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device); checkErr(err);
    *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_device);
    err = clGetDeviceIDs(platforms[N], CL_DEVICE_TYPE_GPU, num_device, *devices, NULL); checkErr(err);
#endif
    //Display devices info
    printf("\nAvailable GPU devices on selected platform (%d found):", num_device);
    for (int i = 0; i < num_device; i++) {
        err = clGetDeviceInfo((*devices)[i], CL_DEVICE_NAME, 0, NULL, &device_name_size); checkErr(err);
        char* device_name = (char*)malloc(sizeof(char) * device_name_size);
        err = clGetDeviceInfo((*devices)[i], CL_DEVICE_NAME, device_name_size, device_name, NULL);
        
        err = clGetDeviceInfo((*devices)[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL); checkErr(err);
        
        printf("\n  [%d] %s (%.1f GB)", i, device_name, (double)mem_size/1000000000);
        free(device_name);
    }
    printf("\nTo select a specific GPU, use --gpu_device <device_id> parameter\n");
}


void SetupContext(cl_platform_id* platforms, cl_device_id* devices, cl_context* context, cl_uint num_device, cl_int gpu_platform_id) {
    cl_int err;
    //choose gpu_platform_id platform
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[gpu_platform_id]), 0 };
    
    // Create context with all devices
    printf("\nCreating OpenCL context with %d device(s)", num_device);
    *context = clCreateContext(properties, num_device, devices, NULL, NULL, &err);
    
    if (err == CL_SUCCESS) {
        // Verify how many devices are actually in the context
        cl_uint num_devices_in_context;
        err = clGetContextInfo(*context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices_in_context, NULL); checkErr(err);
        printf("\nDevices available in context: %d", num_devices_in_context);
        
        // Get the actual device IDs in the context
        size_t device_id_size = num_devices_in_context * sizeof(cl_device_id);
        cl_device_id* context_devices = (cl_device_id*)malloc(device_id_size);
        err = clGetContextInfo(*context, CL_CONTEXT_DEVICES, device_id_size, context_devices, NULL); checkErr(err);
        
        // Show the devices in the context
        for (cl_uint i = 0; i < num_devices_in_context; i++) {
            size_t device_name_size;
            err = clGetDeviceInfo(context_devices[i], CL_DEVICE_NAME, 0, NULL, &device_name_size); checkErr(err);
            char* device_name = (char*)malloc(device_name_size);
            err = clGetDeviceInfo(context_devices[i], CL_DEVICE_NAME, device_name_size, device_name, NULL); checkErr(err);
            printf("\n  Context device %d: %s", i, device_name);
            free(device_name);
        }
        
        free(context_devices);
    } else {
        printf("\nFailed to create context! Error: %d (%s)", err, getErrorString(err));
        exit(-1);
    }
}


void SetupQueue(cl_command_queue* queue, cl_context context, cl_device_id* devices, int device_id) {
    cl_int err = 0;
    cl_command_queue_properties props[] = { CL_QUEUE_PROFILING_ENABLE };
    
    // Get devices from the context (not from the original devices list)
    cl_uint num_devices = 0;
    err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL); checkErr(err);
    
    // Retrieve the context devices
    size_t device_id_size = num_devices * sizeof(cl_device_id);
    cl_device_id* context_devices = (cl_device_id*)malloc(device_id_size);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_id_size, context_devices, NULL); checkErr(err);
    
    // Validate device_id
    if (device_id >= num_devices) {
        printf("\nWarning: Selected GPU device ID %d is out of range (only %d devices available in context). Falling back to device 0.\n", 
               device_id, num_devices);
        device_id = 0;
    }
    
    // Display selected device info
    size_t device_name_size;
    err = clGetDeviceInfo(context_devices[device_id], CL_DEVICE_NAME, 0, NULL, &device_name_size); checkErr(err);
    char* device_name = (char*)malloc(sizeof(char) * device_name_size);
    err = clGetDeviceInfo(context_devices[device_id], CL_DEVICE_NAME, device_name_size, device_name, NULL);
    printf("\nUsing GPU Device %d (from context): %s", device_id, device_name);
    free(device_name);
    
    // Create command queue with selected device from context
    *queue = clCreateCommandQueue(context, context_devices[device_id], *props, &err); checkErr(err);
    
    // Clean up
    free(context_devices);
    
    if (err != CL_SUCCESS) {
        printf("\nFailed to create command queue! Error: %d (%s)", err, getErrorString(err));
        exit(-1);
    }
}


void SetupBuildProgramWithSource(cl_program program_cl, cl_program program_head, cl_device_id* devices, std::string include_path, std::string addtion) {
    cl_int err;
    //const char* input_head_names[1] = { "defines.h" };
    //cl_program input_head[1] = { program_head };
    //  -cl-opt-disable -cl-single-precision-constant -cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros -cl-strict-aliasing
#ifdef LARGE_BOX
    std::string option = " -Werror -cl-single-precision-constant -cl-unsafe-math-optimizations -DLARGE_BOX -cl-mad-enable -cl-std=";
#endif
#ifdef SMALL_BOX
    std::string option = " -Werror -cl-single-precision-constant -cl-unsafe-math-optimizations -DSMALL_BOX -cl-mad-enable -cl-std=";
#endif
#ifdef OPENCL_1_2
    option += std::string("CL1.2");
    printf("\nOpenCL version: 1.2\n");fflush(stdout);
#elif OPENCL_2_0
    option += std::string("CL2.0");
    printf("\nOpenCL version: 2.0\n"); fflush(stdout);
#elif OPENCL_3_0
    option += std::string("CL3.0");
    printf("\nOpenCL version: 3.0\n");fflush(stdout);
#endif
    std::string head_inc_path = "-I ";
    std::string full_path = head_inc_path + include_path + option + addtion;
    const char* options = full_path.data();

    
    //Build program
    err = clBuildProgram(program_cl, 1, devices, options, NULL, NULL);
    if (CL_SUCCESS != err) {
        printf("\nError: Failed to build program executable!");
        char* buffer;
        size_t logsize;
        //Building log
        err = clGetProgramBuildInfo(program_cl, *devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize); checkErr(err);
        buffer = (char*)malloc(logsize * sizeof(char));
        err = clGetProgramBuildInfo(program_cl, *devices, CL_PROGRAM_BUILD_LOG, logsize, buffer, NULL); checkErr(err);
        printf("\nlog:%s", buffer);
        free(buffer);
        exit(-1);
    }
    else {
#ifdef  DISPLAY_SUCCESS
        printf("\nBuild program success!");
        size_t num_kernel;
        err = clGetProgramInfo(program_cl, CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &num_kernel, NULL);
        printf("  Program kernel number: %d\n", (int)num_kernel);
#endif
    }
}

void SaveProgramToBinary(cl_program program_cl,const char* file_name) {
    cl_uint err;
    cl_uint numDevices = 0;
    err = clGetProgramInfo(program_cl, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL); checkErr(err);
    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    err = clGetProgramInfo(program_cl, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL); checkErr(err);
    size_t* programBinarySize = (size_t*)malloc(sizeof(size_t) * numDevices);
    err = clGetProgramInfo(program_cl, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, programBinarySize, NULL); checkErr(err);
    unsigned char** programBinaries = (unsigned char**)malloc(sizeof(unsigned char*) * numDevices);
    for (int i = 0; i < numDevices; i++) {
        programBinaries[i] = (unsigned char*)malloc(sizeof(unsigned char) * programBinarySize[i]);
    }
    err = clGetProgramInfo(program_cl, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, programBinaries, NULL); checkErr(err);
    FILE* fp = fopen(file_name, "w");
    for (int i = 0; i < numDevices; i++) {
        fwrite(programBinaries[i], 1, programBinarySize[i], fp);
    }
    fclose(fp);
}


cl_program SetupBuildProgramWithBinary(cl_context context, cl_device_id* devices, const char* binary_file_name) {
    cl_int err;
    cl_int binary_status;
    
    // Get the number of devices in the context
    cl_uint num_devices = 0;
    err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL); checkErr(err);
    
    // Get the actual device IDs in the context
    size_t device_id_size = num_devices * sizeof(cl_device_id);
    cl_device_id* context_devices = (cl_device_id*)malloc(device_id_size);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_id_size, context_devices, NULL); checkErr(err);
    
    // Load the binary file
    FILE* program_handle = fopen(binary_file_name, "r");
    if (!program_handle) {
        printf("\nError: Could not open binary file: %s", binary_file_name);
        exit(-1);
    }
    
    fseek(program_handle, 0, SEEK_END);
    size_t program_size = ftell(program_handle);
    rewind(program_handle);
    
    char* binary_buffer = (char*)malloc(program_size + 1);
    fread(binary_buffer, sizeof(char), program_size, program_handle);
    binary_buffer[program_size] = '\0';
    fclose(program_handle);
    
    // We need to create the program for each device in the context
    size_t* binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));
    char** binaries = (char**)malloc(num_devices * sizeof(char*));
    
    // Set up the same binary for all devices
    for (cl_uint i = 0; i < num_devices; i++) {
        binary_sizes[i] = program_size;
        binaries[i] = binary_buffer;
    }
    
    // Create the program with the binary for all devices
    cl_program program_cl = clCreateProgramWithBinary(context, num_devices, context_devices, 
                                                     binary_sizes, (const unsigned char**)binaries, 
                                                     NULL, &err); checkErr(err);
    
    // Now build the program for all devices in the context
    printf("\nBuilding OpenCL program for %d device(s)...", num_devices);
    err = clBuildProgram(program_cl, num_devices, context_devices, NULL, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        printf("\nError: Failed to build program for one or more devices (Error: %d - %s)", 
               err, getErrorString(err));
        
        // Print build log for all devices
        for (cl_uint i = 0; i < num_devices; i++) {
            size_t log_size;
            err = clGetProgramBuildInfo(program_cl, context_devices[i], CL_PROGRAM_BUILD_LOG, 
                                       0, NULL, &log_size); checkErr(err);
            
            char* build_log = (char*)malloc(log_size);
            err = clGetProgramBuildInfo(program_cl, context_devices[i], CL_PROGRAM_BUILD_LOG, 
                                       log_size, build_log, NULL); checkErr(err);
            
            // Get device name for better error reporting
            size_t device_name_size;
            err = clGetDeviceInfo(context_devices[i], CL_DEVICE_NAME, 0, NULL, &device_name_size); checkErr(err);
            char* device_name = (char*)malloc(device_name_size);
            err = clGetDeviceInfo(context_devices[i], CL_DEVICE_NAME, device_name_size, device_name, NULL); checkErr(err);
            
            printf("\nBuild log for device %d (%s):\n%s", i, device_name, build_log);
            
            free(device_name);
            free(build_log);
        }
        
        exit(-1);
    } else {
        printf("\nProgram built successfully for all devices");
    }
    
    // Clean up
    free(context_devices);
    free(binary_sizes);
    free(binaries);
    free(binary_buffer);
    
    return program_cl;
}



void SetupKernel(cl_kernel* kernels, cl_program* programs, size_t num_kernel, const char kernel_name[][50]) {
    cl_int err;
    for (int i = 0; i < num_kernel; i++) {
        kernels[i] = clCreateKernel(programs[i], kernel_name[i], &err); checkErr(err);
    }
}

void CreateDeviceBuffer(cl_mem* mem, cl_mem_flags flag, size_t size, cl_context context) {
    cl_int err;
    *mem = clCreateBuffer(context, flag, size, NULL, &err); checkErr(err);
}

void SetKernelArg(cl_kernel kernel, cl_uint num, size_t size, const void *ptr) {
    cl_int err;
    err = clSetKernelArg(kernel, num, size, ptr); checkErr(err);
}