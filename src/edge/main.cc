#include "edge/edge_headers.h"

using namespace intellgraph;

int main(int argc, char** argv) {

  // Initialize Google's logging library.
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 1;
  google::InitGoogleLogging(argv[0]);

  return 0;
}