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
#ifndef INTELLGRAPH_NODE_ACT_LOSS_NODE_H_
#define INTELLGRAPH_NODE_ACT_LOSS_NODE_H_

#include <functional>
#include <vector>
// Your project's .h files
#include "node/output_node.h"
#include "node/node_parameter.h"
#include "utility/common.h"

namespace intellgraph {
// ActLossNode allows user provided function pointers. ActLossNode 
// constructor accepts five parameters: 
// 1. node_param: node paramters
// 2. act_function_ptr: activation function pointer
// 3. act_prime_ptr: activation prime function pointer
// 4. loss_function_ptr: loss function pointer
// 5. loss_prime_ptr: loss function prime pointer
template <class T>
class ActLossNode : public OutputNode<T> {
 public:
  explicit ActLossNode(    
      const NodeParameter& node_param,
      std::function<T(T)> act_function_ptr,
      std::function<T(T)> act_prime_ptr,
      std::function<T(MatXXSPtr<T>, MatXXSPtr<T>)> loss_function_ptr,
      std::function<MatXXSPtr<T>(MatXXSPtr<T>, MatXXSPtr<T>)> loss_prime_ptr);

  ~ActLossNode() {
    std::cout << "ActLossNode " << node_param_.id << " is successfully deleted."
              << std::endl;
  }

  void PrintAct() const;

  void PrintDelta() const;

  void PrintBias() const;

  // Calls activation function and updates activation. Note this function calls 
  // activation function at runtime and thus has performance penalty
  void CallActFxn();

  // Calculates derivative of the activation function and overwrites the 
  // activation in-place. Note this function calls activation prime function at 
  // runtime and thus has performance penalty
  void CalcActPrime();

  void ApplyUnaryFunctor(std::function<T(T)> functor) final;

  // Note this function calls loss function at runtime and thus has performance
  // penalty
  T CalcLoss(MatXXSPtr<T>& data_result) final;

  // Calculates derivative of loss function of weighted_sum variables. Note this
  // function calls loss prime function at runtime and thus has performance
  // penalty
  void CalcDelta(MatXXSPtr<T>& data_result) final;

  inline std::vector<size_t> GetDims() final {
    return node_param_.dims;
  }

  inline MatXXSPtr<T> GetActivationPtr() final {
    return activation_ptr_;
  }

  inline void SetActivationPtr(MatXXSPtr<T>& activation_ptr) final {
    activation_ptr_ = activation_ptr;
    Transition(kInit);
  };

  inline void SetActivation(T value) final {
    activation_ptr_->array() = value;
    Transition(kInit);
  }

  inline MatXXSPtr<T> GetBiasPtr() final {
    return bias_ptr_;
  }

  inline void SetBiasPtr(MatXXSPtr<T>& bias_ptr) final {
    bias_ptr_ = bias_ptr;
  }

  inline MatXXSPtr<T> GetDeltaPtr() final {
    return delta_ptr_;
  }

  inline void SetDeltaPtr(MatXXSPtr<T>& delta_ptr) final {
    delta_ptr_ = delta_ptr;
  }

  inline bool IsActivated() final {
    return current_act_state_ == kAct;
  }

 private:
  void ActToPrime();

  void InitToAct();

  bool Transition(ActStates state);

  const NodeParameter node_param_;
  std::function<T(T)> act_function_ptr_;
  std::function<T(T)> act_prime_ptr_;
  std::function<T(MatXXSPtr<T>, MatXXSPtr<T>)> loss_function_ptr_;
  // Stores derivative of loss function of activation
  std::function<MatXXSPtr<T>(MatXXSPtr<T>, MatXXSPtr<T>)> loss_prime_ptr_;

  MatXXSPtr<T> activation_ptr_;
  // Delta vector stores the derivative of loss function of
  // weighted_sum variables
  MatXXSPtr<T> delta_ptr_;
  MatXXSPtr<T> bias_ptr_;
  // Stores current state of activation vector
  ActStates current_act_state_;
};

template <class T>
using ActLossNodeSPtr = std::shared_ptr<ActLossNode<T>>;

}  // namespace intellgraph

#endif  // INTELLGRAPH_NODE_ACT_LOSS_NODE_H_







  