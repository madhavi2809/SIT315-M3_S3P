#include <stdio.h> // Standard input-output header
#include <stdlib.h> // Standard library header
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h> // OpenCL header
#include <chrono> // Header for timing utilities

#define PRINT 1 // Macro definition for print control

int SZ = 100000000; // Default size for arrays

int *v1, *v2, *v_out; // Pointers for input and output arrays

cl_mem bufV1, bufV2, bufV_out; // OpenCL memory objects

cl_device_id device_id; // OpenCL device ID

cl_context context; // OpenCL context

cl_program program; // OpenCL program

cl_kernel kernel; // OpenCL kernel

cl_command_queue queue; // OpenCL command queue

cl_event event = NULL; // OpenCL event

int err; // Error variable for OpenCL functions

cl_device_id create_device(); // Function prototype for creating OpenCL device

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname); // Function prototype for setting up OpenCL device, context, command queue, and kernel

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename); // Function prototype for building an OpenCL program

void setup_kernel_memory(); // Function prototype for setting up memory buffers for OpenCL kernel arguments

void copy_kernel_args(); // Function prototype for copying arguments to the OpenCL kernel

void free_memory(); // Function prototype for freeing memory allocated for OpenCL objects

void init(int *&A, int size); // Function prototype for initializing an array with random values

void print(int *A, int size); // Function prototype for printing an array

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        SZ = atoi(argv[1]); // Set array size from command line argument
    }

    init(v1, SZ); // Initialize input arrays
    init(v2, SZ);
    init(v_out, SZ);

    size_t global[1] = {(size_t)SZ}; // Define global work size for the kernel

    print(v1, SZ); // Print input arrays
    print(v2, SZ);

    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl"); // Setup OpenCL device, context, command queue, and kernel
    setup_kernel_memory(); // Setup memory buffers for kernel arguments
    copy_kernel_args(); // Copy arguments to the kernel
    auto start = std::chrono::high_resolution_clock::now(); // Start timing kernel execution
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event); // Enqueue kernel for execution
    clWaitForEvents(1, &event); // Wait for kernel execution to finish

    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL); // Read result back from device
    print(v_out, SZ); // Print output array
    auto stop = std::chrono::high_resolution_clock::now(); // Stop timing kernel execution
    std::chrono::duration<double, std::milli> elapsed_time = stop - start; // Calculate elapsed time
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count()); // Print kernel execution time
    free_memory(); // Free allocated memory for OpenCL objects
}

// Function definition for initializing an array with random values
void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size); // Allocate memory for array

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // Assign random values to array elements
    }
}

// Function definition for printing an array
void print(int *A, int size)
{
    if (PRINT == 0) // Check if print control is disabled
    {
        return;
    }

    if (PRINT == 1 && size > 15) // Check if array size is larger than 15
    {
        for (long i = 0; i < 5; i++) // Print first 5 elements
        {
            printf("%d ", A[i]);
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++) // Print last 5 elements
        {
            printf("%d ", A[i]);
        }
    }
    else
    {
        for (long i = 0; i < size; i++) // Print all elements
        {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n"); // Print separator
}

// Function definition for freeing memory allocated for OpenCL objects
void free_memory()
{
    clReleaseMemObject(bufV1); // Release memory objects
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    clReleaseKernel(kernel); // Release kernel
    clReleaseCommandQueue(queue); // Release command queue
    clReleaseProgram(program); // Release program
    clReleaseContext(context); // Release context

    free(v1); // Free input arrays
    free(v2);
    free(v_out);
}

// Function definition for copying arguments to the OpenCL kernel
void copy_kernel_args()
{
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ); // Set kernel arguments
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0) // Check for errors
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

// Function definition for setting up memory buffers for OpenCL kernel arguments
void setup_kernel_memory()
{
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL); // Create memory buffers
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL); // Write input arrays to memory buffers
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Function definition for setting up OpenCL device, context, command queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    // Create command queue using clCreateCommandQueue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    }

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    }
}


// Function definition for building an OpenCL program
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r"); // Open OpenCL program file
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1); // Allocate memory for program buffer
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle); // Read program content into buffer
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err); // Create program from source
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); // Build program
    if (err < 0)
    {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1); // Allocate memory for program log
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL); // Get program build log
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

// Function definition for creating an OpenCL device
cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL); // Get platform ID
    if (err < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL); // Get GPU device ID
    if (err == CL_DEVICE_NOT_FOUND) // Check if GPU not found
    {
        printf("GPU not found\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL); // Get CPU device ID
    }
    if (err < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}
