/* Copyright 2019 The Nicole Authors. All Rights Reserved.
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
#ifndef INTELLGRAPH_EDGE_EDGE_INTERFACE_H_
#define INTELLGRAPH_EDGE_EDGE_INTERFACE_H_

#include "node/node.h"

namespace intellgraph {

template <class T>
interface EdgeInterface {
 public:
  EdgeInterface() noexcept {}

  virtual ~EdgeInterface() noexcept = default;

  virtual void PrintWeight() const = 0;;

  virtual void PrintNablaWeight() const = 0;

  // Calculates weighted sum and updates activation_ptr_ of output layer
  // in-place. Function name with a word 'mute' indicates it requires mutable
  // inputs;
  virtual void Forward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr) = 0;

  // Calculates nabla_weight_ and updates delta_ptr_ of input layer in-place 
  // with backpropagation
  virtual void Backward(MUTE Node<T>* node_in_ptr, MUTE Node<T>* node_out_ptr) = 0;

  // Passes a unary functor and applies it on the weight matrix
  virtual void InitializeWeight(REF const std::function<T(T)>& functor) = 0;

  MUTE virtual inline MatXX<T>* get_weight_ptr() = 0;

  MUTE virtual inline MatXX<T>* get_nabla_weight_ptr() = 0;

  REF virtual inline const MatXX<T>* ref_nabla_weight_ptr() = 0;

};

// Alias for unique Edge Interface pointer
template <class T>
using EdgeIfUPtr = std::unique_ptr<EdgeInterface<T>>;

}  // intellgraph

#endif  // INTELLGRAPH_EDGE_EDGE_INTERFACE_H_