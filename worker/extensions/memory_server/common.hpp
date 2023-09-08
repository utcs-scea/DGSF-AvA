#ifndef __MEMORYSERVER_COMMON_HPP__
#define __MEMORYSERVER_COMMON_HPP__

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>
#include <cstddef>

#define MAX_UUID_LEN 40
namespace GPUMemoryServer {
    inline std::string get_central_socket_path() {
        return std::string("ipc:///tmp/gpumemserver_sock_central");
    }

    enum RequestType { READY, 
        ALLOC, FREE, FINISHED, MEMREQUESTED,
        KERNEL_IN, KERNEL_OUT,
        SCHEDULE };

    union RequestData {
        uint64_t size;
        struct {
            uint32_t port;
        } ready;
    };
    
    struct Request {
        RequestType type;
        RequestData data;
        uint32_t gpu;
        char worker_id[MAX_UUID_LEN]; //36 is the size of a uuid v4
    };

    enum Migration { NOPE, TOTAL, KERNEL};
    enum ReplyCode { OK, RETRY, MIGRATE };
    
    union ReplyData {
        struct {
            Migration type;
            uint32_t target_device;
        } migration;

        struct {
            uint32_t port;
        } ready;
    };

    struct Reply {
        ReplyCode code;
        ReplyData data;
    };

}

#endif