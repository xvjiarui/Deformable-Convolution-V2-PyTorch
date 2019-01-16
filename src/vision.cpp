
#include "deform_psroi_pooling.h"
#include "deform_conv.h"
#include "modulated_deform_conv.h"
#include "deform_conv3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward", &deform_conv_backward, "deform_conv_backward");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "deform_psroi_pooling_forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "deform_psroi_pooling_backward");
  m.def("deform_conv3d_forward", &deform_conv3d_forward, "deform_conv3d_forward");
  m.def("deform_conv3d_backward", &deform_conv3d_backward, "deform_conv3d_backward");
}
