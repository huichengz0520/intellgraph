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
#ifndef INTELLGRAPH_EDGE_EDGE_H_
#define INTELLGRAPH_EDGE_EDGE_H_

#include <functional>

#include "any.h"
#include "edge/edge_interface.h"
#include "edge/edge_parameter.h"
#include "node/node.h"
#include "utility/auxiliary_cpp.h"
#include "utility/common.h"

namespace intellgraph {
// In IntellGraph, edge is a basic building block that is used to connect between 
// two nodes. It is a base class for all edge classes and is implemented with
// static polymorphism.
template <class T, class Instance>
class Edge : public Any<Instance>, implements EdgeInterface<T> {
 public:
  Edge() noexcept {}

  explicit Edge(REF const EdgeParameter& edge_param);

  // Move constructor
  Edge(MOVE Edge<T, Instance>&& rhs) = default;

 // Move operator
  Edge& operator=(MOVE Edge<T, Instance>&& rhs) = default;

  // Copy constructor and operator are explicitly deleted
  Edge(REF const Edge<T, Instance>& rhs) = delete;
  Edge& operator=(REF const Edge<T, Instance>& rhs) = delete;

  virtual ~Edge() noexcept = default;

  void PrintWeight() const final;

  void PrintNablaWeight() const final;

  // Calculates weighted sum and updates activation_ptr_ of output layer
  // in-place. Function name with a word 'mute' indicates it requires mutable
  // inputs;
  void Forward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr) {
    this->ref_instance().Forward(node_in_ptr, node_out_ptr);
  }

  // Calculates nabla_weight_ and updates delta_ptr_ of input layer in-place 
  // with backpropagation
  void Backward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr) {
    this->ref_instance().Backward(node_in_ptr, node_out_ptr);
  }

  // Passes a unary functor and applies it on the weight matrix
  void InitializeWeight(REF const std::function<T(T)>& functor) final;

  MUTE inline MatXX<T>* get_weight_ptr() final {
    return &(this->ref_instance().weight_);
  }

  MUTE inline MatXX<T>* get_nabla_weight_ptr() final {
    return &(this->ref_instance().nabla_weight_);
  }

  REF inline const MatXX<T>* ref_nabla_weight_ptr() final {
    return &(this->ref_instance().nabla_weight_);
  }

 protected:
  EdgeParameter edge_param_{};

  MatXX<T> weight_{};
  MatXX<T> nabla_weight_{};

};

// Alias for unique Edge pointer
template <class T, class Impl>
using EdgeUPtr = std::unique_ptr<Edge<T, Impl>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_H_
