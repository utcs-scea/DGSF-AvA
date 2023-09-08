Install ONNXruntime 1.2.0 from [yuhc/ava-onnxruntime](https://github.com/yuhc/ava-onnxruntime).

Steps to run onnx_opt with a specified dump directory:

0. Set AVA_GUEST_DUMP_DIR and AVA_WORKER_DUMP_DIR on the guestlib application side when running. Two separate envvars are necessary in svless mode due to the guestlib application residing in the VM.

1. Generate the onnx_opt spec with -DAVA_RECV_DUMP_FROM_GUESTLIB in the ava_cxx flags at the top of onnx_opt.cpp. This is to prevent fatbins from being loaded in two separate calls from both the internal_api_handler and the worker. Only the internal_api_handler will load fatbins when it receives the name of the dump directory from the guestlib.

2. Compile ava (without regenerating the spec, so build-ava without the ava-gen dependency) with the addition of the following line to the top of the ava_load_cubin_worker function in `cava/onnx_opt_nw/onnx_opt_nw_worker.cpp`. This is to prevent global metadata values (e.g. number of fatbins) from persisting over multiple applications' loading.
    `ava_metadata_reset(&__ava_endpoint, NULL);`

3. profit

You should see the dump directory name output from the internal_api_handler when the command is received on the worker's side.

