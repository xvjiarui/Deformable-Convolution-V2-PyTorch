
#include "deform_psroi_pooling.h"
#include "deform_conv2d.h"
#include "sparse_conv2d.h"
#include "modulated_deform_conv2d.h"
#include "deform_conv3d.h"
#include "sparse_conv3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv2d_forward", &deform_conv2d_forward, "deform_conv2d_forward");
  m.def("deform_conv2d_backward", &deform_conv2d_backward, "deform_conv2d_backward");
  m.def("modulated_deform_conv2d_forward", &modulated_deform_conv2d_forward, "modulated_deform_conv2d_forward");
  m.def("modulated_deform_conv2d_backward", &modulated_deform_conv2d_backward, "modulated_deform_conv2d_backward");
  m.def("sparse_conv2d_forward", &sparse_conv2d_forward, "sparse_conv2d_forward");
  m.def("sparse_conv2d_backward", &sparse_conv2d_backward, "sparse_conv2d_backward");
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "deform_psroi_pooling_forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "deform_psroi_pooling_backward");
  m.def("deform_conv3d_forward", &deform_conv3d_forward, "deform_conv3d_forward");
  m.def("deform_conv3d_backward", &deform_conv3d_backward, "deform_conv3d_backward");
  m.def("sparse_conv3d_forward", &sparse_conv3d_forward, "sparse_conv3d_forward");
  m.def("sparse_conv3d_backward", &sparse_conv3d_backward, "sparse_conv3d_backward");
}
