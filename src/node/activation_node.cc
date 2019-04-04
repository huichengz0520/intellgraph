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
#include "node/activation_node.h"

namespace intellgraph {

template <class T>
ActivationNode<T>::ActivationNode(const NodeParameter<T>& node_param) {
    node_param_.Clone(node_param);

    size_t row = node_param.get_k_dims()[0];
    size_t col = node_param.get_k_dims()[1];
    
    activation_ptr_ = std::make_unique<MatXX<T>>(row, col);
    delta_ptr_ = std::make_unique<MatXX<T>>(row, col);
    bias_ptr_ = std::make_unique<MatXX<T>>(row, col);

    activation_ptr_->array() = 0.0;
    delta_ptr_->array() = 0.0;
    bias_ptr_->array() = 0.0;

    current_act_state_ = kInit;
}

template <class T>
void ActivationNode<T>::PrintAct() const {
  std::cout << "ActivationNode: " << node_param_.get_k_id() 
            << " Activation Vector:" << std::endl << activation_ptr_->array() 
            << std::endl;
}

template <class T>
void ActivationNode<T>::PrintDelta() const {
  std::cout << "ActivationNode: " << node_param_.get_k_id() << " Delta Vector:" 
            << std::endl << delta_ptr_->array() << std::endl;
}

template <class T>
void ActivationNode<T>::PrintBias() const {
  std::cout << "ActivationNode: " << node_param_.get_k_id() << " Bias Vector:" 
            << std::endl << bias_ptr_->array() << std::endl;
}

template <class T>
void ActivationNode<T>::CallActFxn() {
  if (!Transition(kAct)) {
    std::cout << "ERROR: CallActFxn() for ActivationNode fails" << std::endl;
    exit(1);
  }
}

template <class T>
void ActivationNode<T>::CalcActPrime() {
  if (!Transition(kPrime)) {
    std::cout << "ERROR: CalcActPrime() for ActivationNode fails" << std::endl;
    exit(1);
  }
}

template <class T>
void ActivationNode<T>::ApplyUnaryFunctor_k(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to ApplyUnaryFunctor() is not defined."
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array().unaryExpr(functor);
    Transition(kInit);
  }
}

template <class T>
void ActivationNode<T>::InitializeBias_k(const std::function<T(T)>& functor) {
  if (functor == nullptr) {
    std::cout << "WARNING: functor passed to InitializeBias_k() is not defined."
              << std::endl;
  } else {
    VecX<T> vec(bias_ptr_->array().rows());
    vec.array() = vec.array().unaryExpr(functor);
    bias_ptr_->matrix().colwise() = vec;
    Transition(kInit);
  }
}

// Transitions from kInit state to kAct state. 
template <class T>
void ActivationNode<T>::InitToAct() {
  auto act_functor = node_param_.get_k_act_functor();
  if ( act_functor == nullptr) {
    std::cout << "WARNING: InitToAct() for ActivationNode failed." << std::endl;
    std::cout << "WARNING: activation function is not defined." << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array(). \
                               unaryExpr(act_functor);
  }
  current_act_state_ = kAct;
}

template <class T>
void ActivationNode<T>::ActToPrime() {
  // Derivative equation:
  // $df/dz=f(z)(1-f(z))$
  auto act_prime_functor = node_param_.get_k_act_prime_functor();
  if (act_prime_functor == nullptr) {
    std::cout << "WARNING: ActToPrime() for ActivationNode failed." << std::endl;
    std::cout << "WARNING: activation prime function is not defined."
              << std::endl;
  } else {
    activation_ptr_->array() = activation_ptr_->array(). \
                               unaryExpr(act_prime_functor);
  }
  current_act_state_ = kPrime;
}

template <class T>
bool ActivationNode<T>::Transition(ActStates state) {
  if (state == kInit) {
    current_act_state_ = kInit;
    return true;
  }
  if (current_act_state_ > state) {
    std::cout << "ERROR: Transition() for ActivationNode fails" << std::endl;
    return false;
  }
  while (current_act_state_ < state) {
    switch (current_act_state_) {
      case kInit: {
        InitToAct();
        break;
      }
      case kAct: {
        ActToPrime();
        break;
      }
      default: {
        std::cout << "ERROR: Transition() for ActivationNode fails to handle"
                  << "input state" << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Instantiate class, otherwise compilation will fail
template class ActivationNode<float>;
template class ActivationNode<double>;

}  // namespace intellgraph









