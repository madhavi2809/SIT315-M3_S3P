__kernel void vector_add_ocl(const int size, __global int *v1, __global int *v2, __global int *v_out) {
    const int globalIndex = get_global_id(0); // Get the global index of the current work-item
    
    if (globalIndex < size) { // Check if the global index is within the array size
        // Perform vector addition: Add corresponding elements from v1 and v2 arrays and store the result in v_out array
        v_out[globalIndex] = v1[globalIndex] + v2[globalIndex];
    }
}
