/* Copyright 2019 The IntellGraph Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contributor(s):
	Lingbo Zhang <lingboz2015@gmail.com>
==============================================================================*/
#include "edge/edge.h"

namespace intellgraph {

template <class T>
Edge<T>::Edge(const EdgeParameter& edge_param) {
  edge_param_.Clone(edge_param);

  size_t row = edge_param.ref_dims_in()[0];
  size_t col = edge_param.ref_dims_out()[0];

  weight_ptr_ = std::make_unique<MatXX<T>>(row, col);
  nabla_weight_ptr_ = std::make_unique<MatXX<T>>(row, col);
  
  weight_ptr_->array() = 0.0;
  nabla_weight_ptr_->array() = 0.0;
}

template <class T>
void Edge<T>::PrintWeight() const {
  std::cout << "Edge: " << edge_param_.ref_id() << " Weight matrix:"
            << std::endl << weight_ptr_->array() << std::endl;
}

template <class T>
void Edge<T>::PrintNablaWeight() const {
  std::cout << "Edge: " << edge_param_.ref_id() << " Nabla weight matrix:"
            << std::endl << nabla_weight_ptr_->array() << std::endl;    
}

template <class T>
void Edge<T>::InitializeWeight(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "functor passed to InitializeWeight() is not defined: "
                 << "Initializes weight with standard normal distribution.";
    weight_ptr_->array() = weight_ptr_->array().unaryExpr( \
        std::function<T(T)>(NormalFunctor<T>(0.0, 1.0)));
  } else {
    weight_ptr_->array() = weight_ptr_->array().unaryExpr(functor);
  }
}

// Instantiate class, otherwise compilation will fail
template class Edge<float>;
template class Edge<double>;

}  // intellgraph