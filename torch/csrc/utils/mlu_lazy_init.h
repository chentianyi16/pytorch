#pragma once

#include <c10/core/TensorOptions.h>

namespace torch{
namespace utils{

void mlu_lazy_init();
void set_requires_mlu_init(bool value);

static void maybe_initialize_mlu(const at::TensorOptions& options){
    if(options.device().is_privateuseone()){
        mlu_lazy_init();
    }
}

} // namespace utils
} // namespace torch