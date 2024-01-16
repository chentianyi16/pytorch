#include <torch/csrc/utils/mlu_lazy_init.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch{
namespace utils{
namespace {
    bool is_initialized = false;
}

void mlu_lazy_init(){
    pybind11::gil_scoped_acquire g;
    if (is_initialized){
        return;
    }

    auto module = THPObjectPtr(PyImport_ImportModule("torch.mlu"));
    if(!module){
        throw python_error();
    }

    auto res = THPObjectPtr(PyObject_CallMethod(module.get(),"_lazy_init",""));
    if(!res){
        throw python_error();
    }

    is_initialized = true;
}

void set_requires_mlu_init(bool value){
    is_initialized = !value;
}

}
}