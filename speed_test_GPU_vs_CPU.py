import pyopencl as cl
import numpy as np
import time

# Set the platform
platform = cl.get_platforms()[0]
print(f"Selected Platform: {platform.name}")

# List available devices
print("Available Devices:")
devices = platform.get_devices()
for device in devices:
    print(device)

# Choose specific devices (e.g., the first 1)
selected_devices = platform.get_devices()[0:1]
print(f"Selected Devices: {selected_devices}")

# Create context and program
context = cl.Context(selected_devices)
program = cl.Program(context, """
    __kernel void cube(__global const float *input, __global float *output) {
        int gid = get_global_id(0);
        output[gid] = pow(input[gid], 3);
    }
""").build()

# Generate random input data
input_data = np.random.rand(100000000).astype(np.float32)
output_data = np.empty_like(input_data)

# Copy input data to GPU memory
input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_data.nbytes)

# Measure GPU computation time
start_time = time.time()
queue = cl.CommandQueue(context)
program.cube(queue, input_data.shape, None, input_buffer, output_buffer)
cl.enqueue_copy(queue, output_data, output_buffer).wait()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU Elapsed Time: {elapsed_time} seconds")

# Measure CPU computation time
output_data = np.empty_like(input_data)
start_time = time.time()
output_data = np.power(input_data, 3)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU Elapsed Time: {elapsed_time} seconds")
