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
#ifndef INTELLGRAPH_EDGE_DENSE_EDGE_H_
#define INTELLGRAPH_EDGE_DENSE_EDGE_H_

#include <functional>

#include "glog/logging.h"
#include "edge/edge.h"
#include "edge/edge_parameter.h"
#include "node/node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"
#include "utility/random.h"

namespace intellgraph {

// DenseEdge is an edge class that used to build fully connected neural networks.
// In DenseEdge, weight is updated based on the backpropagation. 
template <class T>
class DenseEdge : public Edge<T, DenseEdge<T>> {
 public:
  DenseEdge() noexcept = default;

  explicit DenseEdge(REF const EdgeParameter& edge_param);
  
  // Move constructor
  DenseEdge(MOVE DenseEdge<T>&& rhs) = default;

  // Move operator
  DenseEdge& operator=(MOVE DenseEdge<T>&& rhs) = default;

  // Copy constructor and operator are explicitly deleted
  DenseEdge(REF const DenseEdge<T>& rhs) = delete;
  DenseEdge& operator=(REF const DenseEdge<T>& rhs) = delete;

  ~DenseEdge() noexcept final = default;

  void Forward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr);

  void Backward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr);

};
// Alias for unique dense edge pointer
template <class T>
using DenseEdgeUPtr = std::unique_ptr<DenseEdge<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_DENSE_EDGE_H_







