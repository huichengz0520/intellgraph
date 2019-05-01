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
#ifndef INTELLGRAPH_EDGE_EDGE_HEADERS_H_
#define INTELLGRAPH_EDGE_EDGE_HEADERS_H_

// Contains edge headers
#include "edge/dense_edge.h"
#include "edge/edge.h"
#include "edge/edge.cc"
#include "edge/edge_factory.h"
#include "edge/edge_interface.h"
#include "edge/edge_parameter.h"
#include "edge/edge_registry.h"

namespace intellgraph {

// Explicit template instantiations
template class Edge<float, DenseEdge<float>>;
template class Edge<double, DenseEdge<double>>;

}

#endif  // INTELLGRAPH_EDGE_EDGE_HEADERS_H_