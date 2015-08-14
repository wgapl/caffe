#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
<<<<<<< HEAD
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
=======
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype alpha = 1.0;
	Dtype beta = 0.0;
	CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			reinterpret_cast<void *>(&alpha),
			bottom_desc_, bottom_data,
			reinterpret_cast<void *>(&beta),
			top_desc_, top_data));
>>>>>>> lrcn/recurrent
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

<<<<<<< HEAD
    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          cudnn::dataType<Dtype>::one,
          top_desc_, top_data, top_desc_, top_diff,
          cudnn::dataType<Dtype>::zero,
=======
    Dtype alpha = 1.0;
    Dtype beta = 0.0;

//    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
//        CUDNN_SOFTMAX_MODE_CHANNEL,
//        top_desc_, top_data, top_desc_, top_diff, bottom_desc_, bottom_diff));

    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          reinterpret_cast<void *>(&alpha),
          top_desc_, top_data, top_desc_, top_diff,
          reinterpret_cast<void *>(&beta),
>>>>>>> lrcn/recurrent
          bottom_desc_, bottom_diff));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif
