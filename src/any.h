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
#ifndef INTELLGRAPH_ANY_H_
#define INTELLGRAPH_ANY_H_

#include "utility/auxiliary_cpp.h"

namespace intellgraph {
// In current Intellgraph, we use the static polymorphism model proposed by 
// Nicolas Burrus, et al.(please refer "A static and complete object-oriented 
// model in C++ mixing benefits of traditional OOP and static programming" in 
// references/ directory.
template <class Instance>
class Any {
 public:
  Any() noexcept = default;

  virtual ~Any() noexcept = default;

  // Gets acutal instance
  MUTE Instance& ref_instance() {
    return *static_cast<Instance*>(this);
  }

};

}  // intellgraph

#endif  // INTELLGRAPH_ANY_H_