from typing import Tuple

from nightwatch.generator.c.stubs import function_implementation
from nightwatch.generator.c.command_handler import handle_command_function, handle_command_header
from nightwatch.model import API, lines


def handle_call(api: API) -> str:
    return handle_command_function(api, list(api.real_functions) + list(api.callback_functions), api.callback_functions)


def source(api: API) -> Tuple[str, str]:
    prelude = f"""
#define __AVA__ 1
#define ava_is_worker 1
#define ava_is_guest 0

#include "worker.h"
#include "worker_context.h"

#undef AVA_BENCHMARKING_MIGRATE

{handle_command_header(api)}

namespace ava {{
WorkerContext::WorkerContext() {{
    __handle_command_{api.identifier.lower()}_init();
    log_file = worker_init_log();
    {api.worker_init_epilogue};
}}
}} //// namespace ava
"""
    stubs = f"""
////// API function stub implementations

#define __chan nw_global_command_channel

{lines(function_implementation(f) for f in api.callback_functions)}
"""

    function = handle_call(api)

    replacement = f"""
////// Worker replacement declarations
#define ava_begin_worker_replacement
#define ava_end_worker_replacement
{api.c_worker_replacement_code}
    """

    return api.c_worker_spelling, prelude + stubs + function + replacement
