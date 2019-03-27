#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<torch/script.h>
#include<stdexcept>
#include<vector>
#include "torch_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

using namespace std;

char *get_and_reset_last_err() {
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

void at_manual_seed(int64_t seed) {
  torch::manual_seed(seed);
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

tensor at_new_tensor() {
  PROTECT(
    return new torch::Tensor();
  )
}

tensor at_tensor_of_data(void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntList(dims, ndims), torch::ScalarType(type));
    if (element_size_in_bytes != tensor.type().elementSizeInBytes())
      throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return new torch::Tensor(tensor);
  )
}

void at_copy_data(tensor tensor, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
    if (elt_size_in_bytes != tensor->type().elementSizeInBytes())
      throw std::invalid_argument("incoherent element sizes in bytes");
    if (numel > tensor->numel())
      throw std::invalid_argument("target numel is larger than tensor numel");
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU);
      void *tensor_data = tmp_tensor.contiguous().data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      void *tensor_data = tensor->contiguous().data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
  )
}

tensor at_shallow_clone(tensor t) {
    PROTECT(return new torch::Tensor(*t);)
}

int at_defined(tensor t) {
  PROTECT(return t->defined();)
}

int at_dim(tensor t) {
  PROTECT(return t->dim();)
}

void at_shape(tensor t, int64_t *dims) {
  PROTECT(
    int i = 0;
    for (int64_t dim : t->sizes()) dims[i++] = dim;
  )
}

int at_scalar_type(tensor t) {
  PROTECT(
    return static_cast<int>(t->scalar_type());
  )
}

int at_device(tensor t) {
  PROTECT(
    return static_cast<int>(t->device().type());
  )
}

void at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(t->backward(c10::nullopt, keep_graph, create_graph);)
}

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
}

int at_grad_set_enabled(int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    return is_enabled;
  )
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
}

template<typename T>
T at_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    return tensor.item<T>();
  )
}

double at_double_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<double>(t, indexes, indexes_len);
}

int64_t at_int64_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template<typename T>
void at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    tensor.fill_(v);
  )
}

void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len, double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len, int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(tensor t, double v) {
  PROTECT(t->fill_(v);)
}

void at_fill_int64(tensor t, int64_t v) {
  PROTECT(t->fill_(v);)
}

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    return strdup(oss.str().c_str());
  )
}

void at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
  )
}

void at_save(tensor t, char *filename) {
  PROTECT(torch::save(*t, filename);)
}

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
  )
}

void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    vector<torch::Tensor> ts(ntensors);
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), ts[i]);
    // Only allocate the new tensor now so that if there is an exception raised during
    // [read], no memory has to be freed.
    for (int i = 0; i < ntensors; ++i)
      tensors[i] = new torch::Tensor(ts[i]);
  )
}

void at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor)) {
  PROTECT(
    shared_ptr<torch::jit::script::Module> module = torch::jit::load(filename);
    if (module == nullptr)
      throw std::invalid_argument("torch::jit::load returned a nullptr");
    for (const auto &p : module->get_parameters()) {
      f(data, (char*)p.key().c_str(), new torch::Tensor(*p.value().slot()));
    }
  )
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i) {
      if (tensors[i]->device().type() == at::kCPU)
        archive.read(std::string(tensor_names[i]), *(tensors[i]));
      else {
        torch::Tensor tmp_tensor = torch::empty_like(*(tensors[i]), at::device(at::kCPU));
        archive.read(std::string(tensor_names[i]), tmp_tensor);
        tensors[i]->copy_(tmp_tensor);
      }
    }
  )
}

tensor at_load(char *filename) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    return new torch::Tensor(tensor);
  )
}

tensor at_load_image(char *filename) {
  PROTECT(
    int w = -1;
    int h = -1;
    int c = -1;
    void *data = stbi_load(filename, &w, &h, &c, 3);
    if (data == nullptr)
      throw std::invalid_argument(stbi_failure_reason());
    torch::Tensor tensor = torch::zeros({ h, w, 3 }, at::ScalarType::Byte);
    memcpy(tensor.data_ptr(), data, h * w * 3);
    free(data);
    return new torch::Tensor(tensor);
  )
}

bool ends_with(const char *str, const char *suffix) {
  int suffix_len = strlen(suffix);
  int str_len = strlen(str);
  if (str_len < suffix_len) return false;
  for (int i = 1; i <= suffix_len; ++i)
    if (str[str_len-i] != suffix[suffix_len-i]) return false;
  return true;
}

int at_save_image(tensor tensor, char *filename) {
  PROTECT(
    auto sizes = tensor->sizes();
    if (sizes.size() != 3)
      throw std::invalid_argument("invalid number of dimensions, should be 3");
    int h = sizes[0];
    int w = sizes[1];
    int c = sizes[2];
    void *tensor_data = tensor->contiguous().data_ptr();
    if (ends_with(filename, ".jpg"))
      return stbi_write_jpg(filename, w, h, c, tensor_data, 90);
    if (ends_with(filename, ".bmp"))
      return stbi_write_bmp(filename, w, h, c, tensor_data);
    if (ends_with(filename, ".tga"))
      return stbi_write_tga(filename, w, h, c, tensor_data);
    return stbi_write_png(filename, w, h, c, tensor_data, 0);
    )
}

tensor at_resize_image(tensor tensor, int out_w, int out_h) {
  PROTECT(
    auto sizes = tensor->sizes();
    if (sizes.size() != 3)
      throw std::invalid_argument("invalid number of dimensions, should be 3");
    int h = sizes[0];
    int w = sizes[1];
    int c = sizes[2];
    const unsigned char *tensor_data = (unsigned char*)tensor->contiguous().data_ptr();
    torch::Tensor out = torch::zeros({ out_h, out_w, c }, at::ScalarType::Byte);
    stbir_resize_uint8(tensor_data, w, h, 0, (unsigned char*)out.data_ptr(), out_w, out_h, 0, c);
    return new torch::Tensor(out);
  )
}

void at_free(tensor t) {
  delete(t);
}

void at_run_backward(tensor *tensors,
                     int ntensors,
                     tensor *inputs,
                     int ninputs,
                     tensor *outputs,
                     int keep_graph,
                     int create_graph) {
  PROTECT(
    torch::autograd::Engine engine;
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::as_variable_ref(*tensors[i]).gradient_edge());

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i)
      inputs_.push_back(torch::autograd::as_variable_ref(*inputs[i]).gradient_edge());

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
  )
}

optimizer ato_adam(double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .beta1(beta1)
        .beta2(beta2)
        .weight_decay(weight_decay);
    return new torch::optim::Adam(vector<torch::Tensor>(), options);
  )
}

optimizer ato_rmsprop(double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered) {
  PROTECT(
    auto options =
      torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered != 0);
      return new torch::optim::RMSprop(vector<torch::Tensor>(), options);
    )
}

optimizer ato_sgd(double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov) {
  PROTECT(
    auto options = 
      torch::optim::SGDOptions(learning_rate)
      .momentum(momentum)
      .dampening(dampening)
      .weight_decay(weight_decay)
      .nesterov(nesterov);
    return new torch::optim::SGD(vector<torch::Tensor>(), options);
  )
}

void ato_add_parameters(optimizer t, tensor *tensors, int ntensors) {
  PROTECT(
    t->add_parameters(of_carray_tensor(tensors, ntensors));
  )
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.learning_rate_ = learning_rate;
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.learning_rate_ = learning_rate;
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.learning_rate_ = learning_rate;
    else
      throw std::invalid_argument("unexpected optimizer");
  )
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.beta1_ = momentum;
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.momentum_ = momentum;
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.momentum_ = momentum;
    else
     throw std::invalid_argument("unexpected optimizer");
  )
}

void ato_zero_grad(optimizer t) {
  PROTECT(t->zero_grad();)
}

void ato_step(optimizer t) {
  PROTECT(t->step();)
}

void ato_free(optimizer t) {
  delete(t);
}

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
}

int64_t ats_to_int(scalar s) {
  PROTECT(return s->toLong();)
}

double ats_to_float(scalar s) {
  PROTECT(return s->toDouble();)
}

char *ats_to_string(scalar s) {
  PROTECT(
    using namespace at;
    std::ostringstream oss;
    oss << (*s);
    return strdup(oss.str().c_str());
  )
}

void ats_free(scalar s) {
  delete(s);
}

int atc_get_num_threads() {
  PROTECT(return at::get_num_threads();)
}

void atc_set_num_threads(int n) {
  PROTECT(at::set_num_threads(n);)
}

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
}

void atc_set_benchmark_cudnn(int b) {
  at::globalContext().setBenchmarkCuDNN(b);
}

module atm_load(char *filename) {
  PROTECT(
    return new std::shared_ptr<torch::jit::script::Module>(torch::jit::load(filename));
  )
}

tensor atm_forward(module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = (*m)->forward(inputs);
    if (!output.isTensor())
      throw std::invalid_argument("forward did not return a tensor");
    return new torch::Tensor(output.toTensor());
  )
}

ivalue atm_forward_(module m,
                    ivalue *ivalues,
                    int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = (*m)->forward(inputs);
    return new torch::jit::IValue(output);
  )
}

void atm_free(module m) {
  delete(m);
}

ivalue ati_tensor(tensor t) {
  PROTECT(
    return new torch::jit::IValue(*t);
  )
}

ivalue ati_int(int64_t i) {
  PROTECT(
    return new torch::jit::IValue(i);
  )
}

ivalue ati_double(double d) {
  PROTECT(
    return new torch::jit::IValue(d);
  )
}

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    return new torch::jit::IValue(torch::jit::Tuple::create(vec));
  )
}

int ati_tag(ivalue i) {
  PROTECT(
    if (i->isTensor()) return 0;
    else if (i->isInt()) return 1;
    else if (i->isDouble()) return 2;
    else if (i->isTuple()) return 3;
    throw std::invalid_argument(("unsupported tag" + i->tagKind()).c_str());
    return -1;
  )
}

int64_t ati_to_int(ivalue i) {
  PROTECT(
    return i->toInt();
  )
}

double ati_to_double(ivalue i) {
  PROTECT(
    return i->toDouble();
  )
}

tensor ati_to_tensor(ivalue i) {
  PROTECT(
    return new torch::Tensor(i->toTensor());
  )
}

int ati_tuple_length(ivalue i) {
  PROTECT(
    return i->toTuple()->elements().size();
  )
}

void ati_to_tuple(ivalue i,
                  ivalue *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toTuple()->elements();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected tuple size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
  )
}

void ati_free(ivalue i) {
  delete(i);
}

#include "torch_api_generated.cpp.h"
