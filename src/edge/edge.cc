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
#ifndef INTELLGRAPH_EDGE_EDGE_CC_
#define INTELLGRAPH_EDGE_EDGE_CC_

#include "edge/edge.h"

namespace intellgraph {

template <class T, class Impl>
Edge<T, Impl>::Edge(const EdgeParameter& edge_param) {
  edge_param_.Clone(edge_param);

  size_t row = edge_param.ref_dims_in()[0];
  size_t col = edge_param.ref_dims_out()[0];

  weight_ = MatXX<T>(row, col);
  nabla_weight_ = MatXX<T>(row, col);

  weight_.array() = 0.0;
  nabla_weight_.array() = 0.0;
}

template <class T, class Impl>
void Edge<T, Impl>::PrintWeight() const {
  std::cout << "Edge: " << edge_param_.ref_id() << " Weight matrix:"
            << std::endl << weight_ << std::endl;
}

template <class T, class Impl>
void Edge<T, Impl>::PrintNablaWeight() const {
  std::cout << "Edge: " << edge_param_.ref_id() << " Nabla weight matrix:"
            << std::endl << nabla_weight_ << std::endl;    
}

template <class T, class Impl>
void Edge<T, Impl>::InitializeWeight(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    LOG(WARNING) << "functor passed to InitializeWeight() is not defined: "
                 << "Initializes weight with standard normal distribution.";
    weight_.array() = weight_.array().unaryExpr( \
        std::function<T(T)>(NormalFunctor<T>(0.0, 1.0)));
  } else {
    weight_.array() = weight_.array().unaryExpr(functor);
  }
}

}  // intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_CC_