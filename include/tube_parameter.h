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
#ifndef NICOLE_TUBE_TUBE_PARAMETER_H_
#define NICOLE_TUBE_TUBE_PARAMETER_H_

#include <vector>

#include "layer/activation_base_layer.h"

namespace nicole {
// TubeParamter contains tube information and is used to construct the tube 
// object
template <class T, class InType, class OutType>
struct TubeParameter {
  size_t id;
  ActBaseLayerSPtr<T, InType> in_layer;
  ActBaseLayerSPtr<T, OutType> out_layer;
};

}  // namespace nicole
#endif  // NICOLE_TUBE_TUBE_PARAMETER_H_